"""
手指和物体粘合: 接触段里手和物体当成一个刚体, 接触的指尖贴在物体点云表面上 (间隙 0, 不打滑)

  python grasp_glue.py --object assets/mouse_pts --hoi-json mouse.json --video data/mouse.mp4 --cache work/mouse_cache.npz

输入: <object>/track.npz (track_object.py), <object>/hand.npz (hand_track.py), HOI json 的 hf 接触链接
输出 <object>/glue.npz:
  obj_R (N, 3, 3), obj_t (N, 3)   物体位姿 (桌面系); 接触段里 = 手+物体刚体的位姿
  kp (N, 21, 3)                   手 21 点 (桌面系); 接触段里固定在物体坐标系里
  contact (N,)                    这一帧在接触段里
  segs (S, 2)                     接触段 [开始, 结束)
  tips_obj (S, 5, 3)              每段 5 个指尖在物体坐标系里的位置; tip_on (S, 5) 哪些指尖贴着物体
<object>/glue.mp4: 物体小球 (红) + 手骨架 (黄, 接触段里青) + 贴着物体的指尖 (绿)

做法 (每个接触段):
  1. 接触段: HOI json 的 hf 链接 (去抖: 间隔 <= 5 帧合并, 短于 10 帧丢掉)
  2. 手掌坐标系: 原点 = 腕 + 4 个指根的平均, x = 小指根 -> 食指根, z = 手掌平面法向
  3. 手 -> 物体的固定相对位姿: 段内 "手检测到 且 物体追踪有效" 的帧, 朝向取平均, 位置取中值
  4. 刚体位姿: 物体追踪有效的帧用追踪; 否则用 手掌坐标系 x 相对位姿; 都没有的帧插值; 时间平滑
  5. 手 21 点: 段内检测到手的帧换到物体坐标系取中值 = 固定的抓握手型;
     中值离物体表面 < TOUCH 的指尖 + 拇指 (接触段 = 抓着, 一定有拇指对握) = 接触指尖, 沿法向投到最近的小球表面上 (间隙 0)
"""
import argparse, json, os, sys
import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from object_points import to_cam, project
from track_object import fill_poses, smooth_poses, read_video
from hand_track import PALM, TIPS

TOUCH = 0.03               # 指尖 (关节中心) 离物体表面 < 3 cm 算贴着 (MediaPipe 指尖点在指腹里面 ~1 cm)
GAP, MIN_LEN = 5, 10       # 接触去抖


def contact_frames(hoi_json, n, min_prob=0.5):
    raw = json.load(open(hoi_json, encoding="utf-8"))
    c = np.zeros(n, bool)
    for fr in raw["frames"]:
        if fr["frame_idx"] < n and any(l["prob"] >= min_prob for l in fr.get("hf", [])):
            c[fr["frame_idx"]] = True
    return c


def segments(c, gap=GAP, min_len=MIN_LEN):
    """接触 0/1 序列 -> [(start, end)]: 合并间隔 <= gap 的段, 丢掉短于 min_len 的段 (同 hoi_to_mujoco.py)"""
    segs, i, n = [], 0, len(c)
    while i < n:
        if c[i]:
            j = i
            while j < n and c[j]:
                j += 1
            if segs and i - segs[-1][1] <= gap:
                segs[-1] = (segs[-1][0], j)
            else:
                segs.append((i, j))
            i = j
        else:
            i += 1
    return [s for s in segs if s[1] - s[0] >= min_len]


def palm_frame(kp):
    """手 21 点 -> 手掌坐标系 (R 列 = x, y, z 轴, 原点)"""
    o = kp[PALM].mean(0)
    x = kp[5] - kp[17]; x /= np.linalg.norm(x)
    z = np.cross(kp[5] - kp[0], kp[17] - kp[0]); z -= (z @ x) * x; z /= np.linalg.norm(z)
    return np.stack([x, np.cross(z, x), z], 1), o


def surface(p, C, r, tree):
    """点 p 到小球拼成的表面: 最近的表面点, 有符号距离 (外正内负)"""
    _, k = tree.query(p)
    v = p - C[k]
    d = np.linalg.norm(v, axis=-1, keepdims=True)
    return C[k] + r * v / np.maximum(d, 1e-9), d[..., 0] - r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", required=True)
    ap.add_argument("--hoi-json", required=True, help="HOI-DETR json (要有 hf 接触链接)")
    ap.add_argument("--video", default=None, help="给了就写 glue.mp4")
    ap.add_argument("--cache", default=None, help="track_object.py 的缓存: 用 SAM2 mask 检查粘合后的位姿")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.object, "meta.json")))
    scene = json.load(open(os.path.join(args.object, "scene.json")))
    tmpl = np.load(os.path.join(args.object, "points.npy")).astype(float)
    r = meta["radius"]
    tree = cKDTree(tmpl)
    tr = np.load(os.path.join(args.object, "track.npz"))
    hd = np.load(os.path.join(args.object, "hand.npz"))
    n = len(tr["t"])
    obj_R, obj_t = tr["R"].copy(), tr["t"].copy()
    kp = hd["kp"].copy()
    ov, hv = tr["valid"].astype(bool), hd["valid"].astype(bool)
    segs = segments(contact_frames(args.hoi_json, n))
    print("接触段 (hf):", segs)

    contact = np.zeros(n, bool)
    tips_all, on_all = [], []
    for s, e in segs:
        contact[s:e] = True
        Rp, op = zip(*[palm_frame(kp[i]) for i in range(s, e)])
        Rp, op = np.array(Rp), np.array(op)
        # 3. 手 -> 物体的相对位姿 (只用两边都是真观测的帧)
        both = [i - s for i in range(s, e) if ov[i] and hv[i]]
        if len(both) < 3:
            raise SystemExit(f"段 {s}-{e}: 手和物体同时看到的帧只有 {len(both)} 帧, 估不出相对位姿")
        rel_R = Rotation.from_matrix([Rp[j].T @ obj_R[s + j] for j in both]).mean().as_matrix()
        rel_t = np.median([Rp[j].T @ (obj_t[s + j] - op[j]) for j in both], 0)
        # 4. 刚体位姿: 物体追踪有效就用它; 否则用 手掌 x 相对位姿 (手背朝相机时 MediaPipe 的手掌朝向误差 2~5 cm, 比物体追踪差)
        R_seg, t_seg, ok = np.zeros((e - s, 3, 3)), np.zeros((e - s, 3)), np.zeros(e - s, bool)
        for j, i in enumerate(range(s, e)):
            if ov[i]:
                R_seg[j], t_seg[j], ok[j] = obj_R[i], obj_t[i], True
            elif hv[i]:
                R_seg[j], t_seg[j], ok[j] = Rp[j] @ rel_R, op[j] + Rp[j] @ rel_t, True
        R_seg, t_seg = smooth_poses(*fill_poses(R_seg, t_seg, ok))
        # 5. 抓握手型: 检测到手的帧换到物体坐标系取中值; 贴着的指尖投到表面上
        hand_obj = np.median([(kp[i] - t_seg[i - s]) @ R_seg[i - s] for i in range(s, e) if hv[i]], 0)
        _, dist = surface(hand_obj[TIPS], tmpl, r, tree)
        on = dist < TOUCH
        on[0] = True                                    # 拇指: 拿起物体必须有拇指对握; 它常被物体挡住, MediaPipe 估得差
        snapped, _ = surface(hand_obj[TIPS], tmpl, r, tree)
        hand_obj[TIPS] = np.where(on[:, None], snapped, hand_obj[TIPS])
        for j, i in enumerate(range(s, e)):
            obj_R[i], obj_t[i] = R_seg[j], t_seg[j]
            kp[i] = hand_obj @ R_seg[j].T + t_seg[j]
        tips_all.append(hand_obj[TIPS]); on_all.append(on)
        jump = [np.linalg.norm(t_seg[0] - tr["t"][s]), np.linalg.norm(t_seg[-1] - tr["t"][min(e, n - 1)])]
        print(f"段 {s}-{e}: 相对位姿用了 {len(both)} 帧; 指尖到表面 (投影前) "
              f"{np.round(dist * 100, 1).tolist()} cm -> 贴着: {[n_ for n_, o in zip(['拇', '食', '中', '无名', '小'], on) if o]}; "
              f"段首/段尾和追踪位姿差 {jump[0] * 100:.1f} / {jump[1] * 100:.1f} cm")

    # 检查: 接触段里贴着的指尖到表面的距离, 手-物体相对运动
    for (s, e), on in zip(segs, on_all):
        g, slip = [], []
        for i in range(s, e):
            C = tmpl @ obj_R[i].T + obj_t[i]
            _, d = surface(kp[i][TIPS][on], C, r, cKDTree(C))
            g.append(np.abs(d).max())
            slip.append(np.abs((kp[i] - obj_t[i]) @ obj_R[i] - (kp[s] - obj_t[s]) @ obj_R[s]).max())
        print(f"段 {s}-{e}: 贴着的指尖到表面 max {max(g) * 1000:.2f} mm, 手相对物体滑动 max {max(slip) * 1000:.2f} mm")

    if args.cache:
        from track_object import Frame
        masks = np.load(args.cache, allow_pickle=True)["masks"]
        for (s, e) in segs:
            idx = [i for i in range(s, e) if masks[i].sum() >= 800]
            # 抓着时物体大半被手挡住, IoU 没意义; 看露出来的 mask 被物体投影盖住的比例
            cover = lambda R, t, i: (lambda fp: (fp & masks[i]).sum() / masks[i].sum())(
                Frame(tmpl, r, scene, np.zeros((0, 3)), masks[i], masks[i], scale=1.0).footprint(R, t))
            a = np.array([cover(obj_R[i], obj_t[i], i) for i in idx])
            b = np.array([cover(tr["R"][i], tr["t"][i], i) for i in idx])
            print(f"段 {s}-{e}: mask >= 800 px 的 {len(idx)} 帧, 露出部分被盖住 粘合后 mean {a.mean():.3f} min {a.min():.3f}"
                  f" vs 只用追踪 mean {b.mean():.3f} min {b.min():.3f}")

    np.savez(os.path.join(args.object, "glue.npz"), obj_R=obj_R, obj_t=obj_t, kp=kp, contact=contact,
             segs=np.array(segs).reshape(-1, 2), tips_obj=np.array(tips_all).reshape(-1, 5, 3),
             tip_on=np.array(on_all).reshape(-1, 5), fps=tr["fps"])
    if args.video:
        import imageio
        rgbs, fps = read_video(args.video)
        bones = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
                 (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
        seg_of = {i: k for k, (s, e) in enumerate(segs) for i in range(s, e)}
        frames = []
        for i in range(n):
            img = cv2.cvtColor(rgbs[i], cv2.COLOR_RGB2BGR)
            for u, v in np.round(project(to_cam(tmpl @ obj_R[i].T + obj_t[i], scene), scene)).astype(int):
                cv2.circle(img, (int(u), int(v)), 1, (0, 0, 255), -1)
            q = np.round(project(to_cam(kp[i], scene), scene)).astype(int)
            for a, b in bones:
                cv2.line(img, tuple(q[a]), tuple(q[b]), (255, 255, 0) if contact[i] else (0, 255, 255), 1)
            if i in seg_of:
                for k, on in zip(TIPS, on_all[seg_of[i]]):
                    if on:
                        cv2.circle(img, tuple(q[k]), 4, (0, 255, 0), -1)
            cv2.putText(img, f"{i}{' contact' if contact[i] else ''}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        imageio.mimsave(os.path.join(args.object, "glue.mp4"), frames, fps=fps, macro_block_size=1)
    print("saved", os.path.join(args.object, "glue.npz"))


if __name__ == "__main__":
    main()
