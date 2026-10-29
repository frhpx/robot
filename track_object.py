"""
点云追踪物体: 每帧物体在桌面坐标系里的 6D 位姿 (位置 + 朝向), 给 hoi_to_mujoco.py --obj-track 用

  python track_object.py --video data/mouse.mp4 --object assets/mouse_pts
  python track_object.py ... --cache work/mouse_cache.npz     # SAM2 / MoGe / 手 mask 存起来, 再跑只算位姿

输入: object_points.py 生成的资产 (points.npy 模板, scene.json 桌面系 + 内参 + 参考帧, mask.png 参考帧 mask)
输出 <object>/track.npz:
  R (N, 3, 3), t (N, 3)  物体系 -> 桌面系: p_table = R @ p_obj + t  (t = 物体包围盒中心, 和 points.npy 原点一致)
  valid (N,)             这一帧看得到物体、位姿可信 (无效帧的位姿是前后有效帧插值的)
  n_obs, rmse, iou, cover, conflict, scale, mode (0 = 平放 x/y/yaw, 1 = 6D)
<object>/track.mp4: 绿 = SAM2 mask 轮廓, 蓝 = 手 mask 轮廓, 红 = 追踪到的小球中心投影

流程:
  1. SAM2 视频分割: 参考帧的 mask.png 作提示, 向前 / 向后传播 -> 每帧物体 mask
     MediaPipe 手部 21 点的凸包 (膨胀) = 手 mask (遮挡物)
  2. MoGe-2 点图 (fov 固定 = scene.json). 相机不动, MoGe 每帧尺度会抖: 用物体周围一圈桌面 (去掉手) 的
     参考帧深度 / 这一帧深度 的截尾中值, 把这一帧缩放到参考帧尺度
  3. mask 内 (腐蚀 3 px) 的点 -> 桌面系, 去掉桌面 3 mm 以内的点和离群点
  4. 位姿 = 最小化 3D 项 + 剪影项, 从上一帧出发 (多个初值, Nelder-Mead):
       3D:   观测点到模板小球表面的距离 (截尾 RMS)
       剪影: mask 里没被物体投影盖住的比例 + 物体投影落在 "看得到的背景" 上的比例 x 2
             (看得到的背景 = 不是物体 mask、不是手、深度不比物体近; 被手挡住的地方两边都不算)
     桌面和相机已知, 物体平放时剪影本身就定下 x / y / yaw, 3D 项消除对称歧义;
     只有观测点高过物体平放时够得到的高度、且 6D 残差明显更小时才用 6D;
     另加一个弱的时间先验 (和上一帧的差), 只露出一小条时位姿不会在手底下乱漂
  5. mask 太小 (被手整个挡住) 或代价太大 -> 无效, 用前后有效帧插值; 最后时间平滑
"""
import argparse, json, os, sys, tempfile
import numpy as np
import cv2
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from object_points import to_table, to_cam, project, remove_outliers, MIN_Z

MIN_OBS = 40                # 少于这么多观测点 -> 不用 3D 项
MIN_MASK = 150              # mask 少于这么多像素 -> 这一帧无效 (物体基本被挡住)
MAX_COST = 0.6              # 代价超过 -> 无效
E3D = 0.003                 # 3D 项的尺度: 残差 3 mm 记 1
W3D = 0.3                   # 3D 项权重
W_CONFLICT = 2.0            # 投影落在看得到的背景上的权重
LIFT_OBS = 0.01             # 观测点 90% 分位高度超过 物体高度 + 这么多 (平放时够不到) 才考虑 6D
W_PRIOR = 0.05              # 和上一帧的差: 2 cm / 20° 记 W_PRIOR (只露出一小条时, 不让位姿在手底下乱漂)
MAX_JUMP = 0.03             # 和上一有效帧相比 1 帧内跳变超过 3 cm -> 无效


# ================= 1. 分割: 物体 (SAM2) + 手 (MediaPipe) =================
def read_video(video):
    cap = cv2.VideoCapture(video)
    rgbs = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgbs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return rgbs, cap.get(cv2.CAP_PROP_FPS)


def video_masks(rgbs, ref, ref_mask, model_id="facebook/sam2.1-hiera-small"):
    """参考帧 mask 向两个方向传播, 返回 (n, H, W) bool"""
    import torch
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    tmp = tempfile.mkdtemp(prefix="sam2_")
    for i, rgb in enumerate(rgbs):
        cv2.imwrite(os.path.join(tmp, f"{i:05d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    pred = SAM2VideoPredictor.from_pretrained(model_id)
    masks = np.zeros((len(rgbs), *ref_mask.shape), bool)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        state = pred.init_state(video_path=tmp, offload_video_to_cpu=True)
        pred.add_new_mask(state, frame_idx=ref, obj_id=1, mask=ref_mask)
        for reverse in (False, True):
            for i, _, logits in pred.propagate_in_video(state, start_frame_idx=ref, reverse=reverse):
                masks[i] = (logits[0, 0] > 0).cpu().numpy()
    del pred; torch.cuda.empty_cache()
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    return np.array([largest_component(m) for m in masks])


def largest_component(m):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    if n <= 2:
        return m
    return lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])


def hand_masks(rgbs, grow=12):
    """MediaPipe 手部关键点的凸包, 膨胀 grow px; 视频模式 (跟踪), 最多两只手"""
    import mediapipe as mp
    H, W = rgbs[0].shape[:2]
    out = np.zeros((len(rgbs), H, W), bool)
    with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3,
                                  min_tracking_confidence=0.3) as hands:
        for i, rgb in enumerate(rgbs):
            res = hands.process(rgb)
            m = np.zeros((H, W), np.uint8)
            for lm in res.multi_hand_landmarks or []:
                xy = np.array([[p.x * W, p.y * H] for p in lm.landmark], np.int32)
                cv2.fillConvexPoly(m, cv2.convexHull(xy), 1)
            out[i] = dilate(m, grow)
    return out


def dilate(m, px):
    return cv2.dilate(m.astype(np.uint8), np.ones((2 * px + 1, 2 * px + 1), np.uint8)).astype(bool)


# ================= 2. MoGe 点图 =================
def moge_frames(rgbs, fov_x, masks, model_id="Ruicheng/moge-2-vitl-normal"):
    """每帧: 深度图 z (无效 = NaN, float32) 和 mask 核心区域 (腐蚀 3 px) 的相机系点"""
    import torch
    from moge.model.v2 import MoGeModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoGeModel.from_pretrained(model_id).to(dev).eval()
    zs, obs = [], []
    for i, rgb in enumerate(rgbs):
        img = torch.tensor(rgb / 255, dtype=torch.float32, device=dev).permute(2, 0, 1)
        with torch.no_grad():
            out = model.infer(img, fov_x=fov_x, use_fp16=(dev == "cuda"))
        pts, valid = out["points"].cpu().numpy(), out["mask"].cpu().numpy().astype(bool)
        zs.append(np.where(valid, pts[..., 2], np.nan).astype(np.float32))
        core = cv2.erode(masks[i].astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & valid
        obs.append(pts[core].astype(np.float32))
        if i % 50 == 0:
            print(f"MoGe {i}/{len(rgbs)}")
    del model; torch.cuda.empty_cache()
    return np.array(zs), obs


def local_scale(z, z_ref, region):
    """参考帧深度 / 这一帧深度, 在 region 里取截尾中值 (去掉偏离中值 > 5% 的, 比如漏掉的手)"""
    r = z_ref[region] / z[region]
    r = r[np.isfinite(r) & (r > 0)]
    if len(r) < 300:
        return 1.0
    s = np.median(r)
    for _ in range(3):
        s = np.median(r[np.abs(r / s - 1) < 0.05])
    return float(s)


# ================= 3. 位姿代价 =================
def pose_from(x, planar, z_rest):
    """planar: x = (x, y, yaw); 否则 x = (x, y, z, 旋转向量)"""
    if planar:
        return Rotation.from_euler("z", x[2]).as_matrix(), np.array([x[0], x[1], z_rest])
    return Rotation.from_rotvec(x[3:]).as_matrix(), np.asarray(x[:3], float)


def params_from(R, t, planar):
    if planar:
        return np.array([t[0], t[1], Rotation.from_matrix(R).as_euler("zyx")[0]])
    return np.r_[t, Rotation.from_matrix(R).as_rotvec()]


class Frame:
    """一帧的观测: 3D 点 (桌面系), 物体 mask, 看得到的背景, 用来算代价"""

    def __init__(self, tmpl, r, scene, obs, mask, free, scale=0.5, prior=None):
        self.tmpl, self.r, self.scene, self.obs, self.prior = tmpl, r, scene, obs, prior
        self.k = scale                                 # 剪影在半分辨率上算
        sz = (int(mask.shape[1] * scale), int(mask.shape[0] * scale))
        self.mask = cv2.resize(mask.astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST).astype(bool)
        self.free = cv2.resize(free.astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST).astype(bool)
        self.area = max(1, self.mask.sum())
        self.K = np.array(scene["K"]) * [[scene["width"]], [scene["height"]], [1]]

    def footprint(self, R, t):
        pc = to_cam(self.tmpl @ R.T + t, self.scene)
        uv = project(pc, self.scene) * self.k
        rad = np.maximum(1, np.round(self.K[0, 0] * self.r / pc[:, 2] * self.k)).astype(int)
        fp = np.zeros(self.mask.shape, np.uint8)
        for (u, v), q in zip(np.round(uv).astype(int), rad):
            cv2.circle(fp, (int(u), int(v)), int(q), 1, -1)
        return fp.astype(bool)

    def e3d(self, R, t):
        if len(self.obs) < MIN_OBS:
            return 0.0
        d, _ = cKDTree(self.tmpl @ R.T + t).query(self.obs)
        return float(np.sqrt(np.mean(np.minimum(np.abs(d - self.r), 0.01) ** 2)))

    def terms(self, R, t):
        fp = self.footprint(R, t)
        miss = (self.mask & ~fp).sum() / self.area
        conflict = (fp & self.free).sum() / max(1, fp.sum())
        return self.e3d(R, t), miss, conflict

    def cost(self, R, t):
        e, miss, conflict = self.terms(R, t)
        c = W3D * (e / E3D) ** 2 + miss + W_CONFLICT * conflict
        if self.prior is not None:
            Rp, tp = self.prior
            ang = Rotation.from_matrix(R @ Rp.T).magnitude()
            c += W_PRIOR * ((np.linalg.norm(t - tp) / 0.02) ** 2 + (ang / np.radians(20)) ** 2)
        return c


def optimize(fr, inits, planar, z_rest):
    """多个初值各跑一次 Nelder-Mead, 返回代价最小的 (R, t, cost)"""
    step = np.array([0.006, 0.006, np.radians(10)]) if planar else np.array([0.006, 0.006, 0.006, 0.1, 0.1, 0.1])
    best = None
    for R0, t0 in inits:
        x0 = params_from(R0, t0, planar)
        sim = np.vstack([x0, x0 + np.diag(step)])
        res = minimize(lambda x: fr.cost(*pose_from(x, planar, z_rest)), x0, method="Nelder-Mead",
                       options=dict(initial_simplex=sim, xatol=1e-4, fatol=1e-4, maxfev=300 if planar else 600))
        if best is None or res.fun < best[2]:
            best = (*pose_from(res.x, planar, z_rest), float(res.fun))
    return best


def bottom(R, t, tmpl, r):
    """模板最低点 (含小球半径) 在桌面系的高度"""
    return float((tmpl @ R.T + t)[:, 2].min() - r)


# ================= 4. 平滑 / 插值 =================
def fill_poses(R, t, valid):
    """无效帧: 位置线性插值, 朝向 slerp; 首尾外推为最近的有效帧"""
    idx = np.where(valid)[0]
    n = len(valid)
    tt = np.stack([np.interp(np.arange(n), idx, t[idx, c]) for c in range(3)], 1)
    rot = Slerp(idx, Rotation.from_matrix(R[idx]))(np.clip(np.arange(n), idx[0], idx[-1]))
    return rot.as_matrix(), tt


def smooth_poses(R, t, win=5):
    """位置: 5 帧中值 + win 帧滑动平均; 朝向: win 帧窗口内四元数平均"""
    n = len(t)
    p = np.pad(t, ((2, 2), (0, 0)), mode="edge")
    t = np.median(np.stack([p[i:i + n] for i in range(5)]), 0)
    k = np.ones(win) / win
    t = np.stack([np.convolve(np.pad(t[:, c], win // 2, mode="edge"), k, "valid") for c in range(3)], 1)
    rots = Rotation.from_matrix(R)
    return np.array([rots[max(0, i - win // 2):min(n, i + win // 2 + 1)].mean().as_matrix() for i in range(n)]), t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--object", required=True, help="object_points.py 的输出目录 (要有 scene.json)")
    ap.add_argument("--cache", default=None, help="SAM2 / MoGe / 手 mask 的缓存 npz: 有就读, 没有就算完写进去")
    ap.add_argument("--out", default=None, help="默认 <object>/track.npz")
    ap.add_argument("--no-video", action="store_true", help="不写 track.mp4")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.object, "meta.json")))
    scene = json.load(open(os.path.join(args.object, "scene.json")))
    tmpl = np.load(os.path.join(args.object, "points.npy")).astype(float)
    r = meta["radius"]
    ref = scene["ref_frame"]
    ref_mask = cv2.imread(os.path.join(args.object, "mask.png"), 0) > 0
    rgbs, fps = read_video(args.video)
    n = len(rgbs)
    if rgbs[0].shape[:2] != (scene["height"], scene["width"]):
        raise SystemExit(f"视频 {rgbs[0].shape[1]}x{rgbs[0].shape[0]} 和 scene.json {scene['width']}x{scene['height']} 不一致")

    if args.cache and os.path.exists(args.cache):
        c = np.load(args.cache, allow_pickle=True)
        masks, hands, zs, obs_cam = c["masks"], c["hands"], c["zs"], list(c["obs"])
        print("读缓存", args.cache)
    else:
        print(f"SAM2: 参考帧 {ref} 的 mask 传播到 {n} 帧")
        masks = video_masks(rgbs, ref, ref_mask)
        hands = hand_masks(rgbs)
        zs, obs_cam = moge_frames(rgbs, scene["fov_x"], masks)
        if args.cache:
            os.makedirs(os.path.dirname(os.path.abspath(args.cache)), exist_ok=True)
            np.savez(args.cache, masks=masks, hands=hands, zs=zs, obs=np.array(obs_cam, dtype=object))

    center0 = np.array(meta["pose_table"]) + [0, 0, meta["size"][2] / 2]   # 参考帧: 包围盒中心 (桌面系)
    z_rest = center0[2]
    ring = dilate(ref_mask, 70) & ~dilate(ref_mask, 15)
    ring[:np.where(ref_mask)[0].min()] = False          # 物体顶边以上是墙, 不用
    keys = ("R", "t", "valid", "n_obs", "rmse", "miss", "conflict", "cost", "scale", "mode")
    res = {}
    R, t = np.eye(3), center0.copy()
    lost = 0                                            # 距离上一个有效帧多少帧
    for i in list(range(ref, n)) + list(range(ref - 1, -1, -1)):   # 从参考帧向后, 再从参考帧向前
        if i == ref - 1:
            R, t, lost = res[ref][0], res[ref][1], 0
        m, hd = masks[i], hands[i]
        region = ring & ~dilate(m, 15) & ~hd & np.isfinite(zs[i]) & np.isfinite(zs[ref])
        s = local_scale(zs[i], zs[ref], region)
        obs = to_table(obs_cam[i] * s, scene) if len(obs_cam[i]) else np.zeros((0, 3))
        obs = obs[obs[:, 2] > MIN_Z]
        if len(obs) >= MIN_OBS:
            obs, _ = remove_outliers(obs, obs)
        zc = to_cam(t[None], scene)[0, 2]               # 上一帧物体中心的深度
        free = ~dilate(m, 2) & ~hd & ~(zs[i] * s < zc - 0.03)   # 比物体近 3 cm 以上的像素是遮挡物
        fr = Frame(tmpl, r, scene, obs, m, free, prior=(R, t) if not lost else None)
        ok = m.sum() >= MIN_MASK
        Ri, ti, cost, mode = R, t, np.inf, 0
        if ok:
            flat = lambda Rx: Rotation.from_euler("z", Rotation.from_matrix(Rx).as_euler("zyx")[0]).as_matrix()
            yaw = [Rotation.from_euler("z", a, degrees=True).as_matrix() @ flat(R) for a in (0, -20, 20)]
            Ri, ti, cost = optimize(fr, [(Ry, np.r_[t[:2], z_rest]) for Ry in yaw], True, z_rest)
            if (cost > 0.3 or lost) and len(obs) >= MIN_OBS:   # 跟丢 / 解释不好: 从观测点中心重新找, 8 个朝向
                c = np.median(obs, 0)
                inits = [(Rotation.from_euler("z", a, degrees=True).as_matrix(), np.r_[c[:2], z_rest])
                         for a in range(0, 360, 45)]
                Rg, tg, cg = optimize(fr, inits, True, z_rest)
                if cg < cost:
                    Ri, ti, cost = Rg, tg, cg
            e_flat = fr.e3d(Ri, ti)
            if len(obs) >= MIN_OBS and np.percentile(obs[:, 2], 90) > meta["size"][2] + LIFT_OBS and e_flat > E3D:
                # 观测点离开桌面且平放解释不了: 试 6D (从上一帧 / 平放解 / 观测点中心出发)
                R6, t6, c6 = optimize(fr, [(R, t), (Ri, ti), (Ri, np.median(obs, 0))], False, z_rest)
                if fr.e3d(R6, t6) < 0.7 * e_flat and c6 < cost:
                    Ri, ti, cost, mode = R6, t6, c6, 1
            ok = cost < MAX_COST and np.linalg.norm(ti - t) < MAX_JUMP * (1 + lost)
        e, miss, conflict = fr.terms(Ri, ti)
        if ok:
            R, t, lost = Ri, ti, 0
        else:
            lost += 1
        res[i] = (Ri, ti, ok, len(obs), e, miss, conflict, cost, s, mode)
        if i % 25 == 0:
            print(f"frame {i}: mask {m.sum()} px, 观测 {len(obs)} 点, 3D {e * 1000:.1f} mm, 漏盖 {miss:.2f}, "
                  f"冲突 {conflict:.2f}, 尺度 {s:.3f}, {'6D' if mode else '平放'}, 中心 {np.round(ti, 3).tolist()}"
                  f"{'' if ok else '  (无效)'}")

    out = {k: np.array([res[i][j] for i in range(n)]) for j, k in enumerate(keys)}
    valid = out["valid"].astype(bool)
    if valid.sum() < 2:
        raise SystemExit("有效帧太少, 检查 SAM2 mask / MoGe")
    Rf, tf = smooth_poses(*fill_poses(out["R"], out["t"], valid))

    # 检查: 小球投影 vs SAM2 mask. IoU 只在手没碰到物体的帧上有意义; 被挡住的帧看 cover (mask 被盖住的比例)
    iou, cover, occl = np.zeros(n), np.zeros(n), np.zeros(n, bool)
    frames = []
    for i in range(n):
        fr = Frame(tmpl, r, scene, np.zeros((0, 3)), masks[i], np.zeros_like(masks[i]), scale=1.0)
        fp, mk = fr.footprint(Rf[i], tf[i]), masks[i]
        iou[i] = (fp & mk).sum() / max(1, (fp | mk).sum())
        cover[i] = (fp & mk).sum() / max(1, mk.sum())
        occl[i] = (hands[i] & dilate(fp, 5)).any()
        if not args.no_video:
            img = cv2.cvtColor(rgbs[i], cv2.COLOR_RGB2BGR)
            uv = project(to_cam(tmpl @ Rf[i].T + tf[i], scene), scene)
            for u, v in np.round(uv).astype(int):
                cv2.circle(img, (int(u), int(v)), 1, (0, 0, 255), -1)
            for mm, col in ((mk, (0, 255, 0)), (hands[i], (255, 128, 0))):
                cs, _ = cv2.findContours(mm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, cs, -1, col, 1)
            cv2.putText(img, f"{i} IoU {iou[i]:.2f}{' 6D' if out['mode'][i] else ''}{'' if valid[i] else ' interp'}",
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    path = args.out or os.path.join(args.object, "track.npz")
    np.savez(path, R=Rf, t=tf, valid=valid, iou=iou, cover=cover, occluded=occl, fps=fps, ref_frame=ref,
             R_raw=out["R"], t_raw=out["t"], **{k: out[k] for k in keys if k not in ("R", "t", "valid")})
    if frames:
        import imageio
        imageio.mimsave(os.path.join(args.object, "track.mp4"), frames, fps=fps, macro_block_size=1)
    lift = np.array([bottom(Rf[i], tf[i], tmpl, r) for i in range(n)])
    step = np.linalg.norm(np.diff(tf, axis=0), axis=1)
    full = masks.reshape(n, -1).sum(1) >= 0.8 * ref_mask.sum()          # 物体基本没被挡: mask >= 参考帧的 80%
    print(f"有效帧 {valid.sum()}/{n}, 6D 帧 {int(out['mode'].sum())}, 尺度 {out['scale'].min():.3f}~{out['scale'].max():.3f}")
    print(f"IoU (没被挡的 {full.sum()} 帧): mean {iou[full].mean():.3f}, min {iou[full].min():.3f} "
          f"(第 {np.where(full)[0][iou[full].argmin()]} 帧)")
    print(f"mask 覆盖 (有 mask 的帧): mean {cover[masks.reshape(n, -1).any(1)].mean():.3f}; "
          f"最高抬起 {lift.max() * 100:.1f} cm, 每帧位移 max {step.max() * 1000:.1f} mm")
    print("saved", path)


if __name__ == "__main__":
    main()
