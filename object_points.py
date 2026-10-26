"""
从视频的参考帧生成点云物体资产 (小球拼成的刚体), 并用桌面平面估计相机姿态; hoi_to_mujoco.py --object 直接用

  python object_points.py --video data/mouse.mp4 --hoi-json data/hoi/mouse.json --hoi-repo ~/code/handbench/hoi \
      --name mouse_pts --depth-npz data/mouse_moge.npz
  python object_points.py ... --ref-frame 20       # 指定参考帧 (默认自动挑: 没有手 / 右手离物体最远)
  python object_points.py ... --fill column        # 不对称的物体: 竖直补柱子 (默认 mirror 对称镜像)

--depth-npz (moge_depth.py 的输出) 用来固定 fov_x, 让物体/桌面和手轨迹的尺度一致; 不给就让 MoGe 自己估 fov

输出 assets/<name>/:
  points.npy   (N, 3) 物体坐标系 (轴和桌面坐标系一致, 保留物体在桌上的实际朝向), 米, 原点在包围盒中心
  colors.npy   (N, 3) 每个点的颜色 0~1
  meta.json    type="points", n, radius, size (包围盒尺寸, 含小球半径), dims_obb (沿俯视主轴的 长/宽/高),
               mass, ref_frame, fill, voxel, pose_table (物体底面中心在桌面坐标系里的位置)
  scene.json   R_cam2table, p_table (桌面上一点, 相机系), ref_frame, fov_x, K (归一化内参), 图像尺寸, pitch_deg
  mask.png     参考帧分割结果, preview.png 俯视/正视/侧视图

桌面坐标系: 原点 = 桌面上一点, +z = 桌面法向 (朝相机一侧), +x = 相机视线在桌面上的投影, +y = z × x
  p_table = R_cam2table @ (p_cam - p_table0)

流程:
  1. 参考帧: 物体静止、手离得远的一帧 (默认: 没有任何手框的帧里离第一次检测到物体最近的; 都有手就取右手离物体最远的)
     参考帧没有物体框时, 借用 pick_frame (物体最清楚的那帧) 的框作 SAM2 提示 (物体在被碰之前不动)
  2. MoGe-2 点图; 物体 mask 和手框以外、物体顶边以下的有效点 RANSAC 拟合平面 (5 mm, 内点 >= 30%) = 桌面
  3. SAM2 分割 -> mask 腐蚀 3 px -> MoGe 点 -> 桌面坐标系; 去离群点 (中位距离 + 3 MAD), 去掉桌面上 3 mm 以内的点
  4. 补全 (mirror): 俯视 PCA 长轴; 对称面过长轴、垂直桌面, 沿短轴搜位置: 镜像点和观测点的 Chamfer 距离 + 冲突惩罚
     (镜像点投影回参考帧, 比该像素的 MoGe 深度更靠近相机 = 落进相机看得到的空地); 合并后加一层底面 (俯视凸包)
     (column): 俯视凸包补顶面, 再竖直补到桌面
  5. 体素降采样 (默认 7 mm), N 控制在 150~500; 小球半径 r = 0.6 voxel, 最低的小球底面贴桌面
"""
import argparse, json, os, sys
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from object_asset import pick_frame, read_frame, segment

MAX_N, MIN_N = 500, 150
R_FACTOR = 0.6              # 小球半径 / 体素边长
MIN_Z = 0.003               # 离桌面 3 mm 以内的观测点当成桌面 / 阴影, 不要


# ================= 1. 参考帧 =================
def gap(a, b):
    """两个框 (xyxy) 之间的最短距离 (像素), 重叠为 0"""
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return np.hypot(dx, dy)


def choose_ref_frame(fb, obj_box):
    """第一次检测到物体之前 (物体还没被碰过, 位姿和接触开始时一样), 最长的一段 "没有任何手框" 的帧的中间一帧
       (段的两头常有检测漏掉的、刚进画面的手); 每帧都有手就取右手框离物体框最远的一帧"""
    first = next((f.idx for f in fb if f.obj is not None), len(fb))
    free = [not f.hands and f.left is None and f.right is None for f in fb][:first]
    best, i = None, 0
    while i < len(free):
        if free[i]:
            j = i
            while j < len(free) and free[j]:
                j += 1
            if best is None or j - i > best[1] - best[0]:
                best = (i, j)
            i = j
        else:
            i += 1
    if best:
        return (best[0] + best[1]) // 2
    return max(range(max(1, first)), key=lambda i: np.inf if fb[i].right is None else gap(fb[i].right, obj_box))


def prompt_box(fb, i):
    """SAM2 提示框: 这一帧有物体框就用, 没有就借物体最清楚的那一帧的框"""
    if fb[i].obj is not None:
        return fb[i].obj
    j = pick_frame(fb)
    if j is None:
        raise SystemExit("整段视频都没有物体框")
    return fb[j].obj


# ================= 2. MoGe + 桌面 =================
def moge_infer(rgb, fov_x=None, model_id="Ruicheng/moge-2-vitl-normal"):
    """MoGe-2: 点图 (H, W, 3, 相机系 x右 y下 z前) + 有效掩码 + 归一化内参 + fov_x (度)"""
    import torch
    from moge.model.v2 import MoGeModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoGeModel.from_pretrained(model_id).to(dev).eval()
    img = torch.tensor(rgb / 255, dtype=torch.float32, device=dev).permute(2, 0, 1)
    with torch.no_grad():
        out = model.infer(img, fov_x=fov_x, use_fp16=(dev == "cuda"))
    K = out["intrinsics"].cpu().numpy().astype(float)
    pts, valid = out["points"].cpu().numpy(), out["mask"].cpu().numpy().astype(bool)
    del model; torch.cuda.empty_cache()
    return pts, valid, K, float(np.degrees(2 * np.arctan(0.5 / K[0, 0])))


def fit_plane(p, thr=0.005, iters=1000, up=np.array([0, -1.0, 0]), max_tilt=60, seed=0):
    """RANSAC 平面 (只考虑法向和相机 "上" 方向夹角 < max_tilt 的平面, 排除墙面); 返回 (法向, 平面上一点, 内点掩码)"""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if abs(n @ up) < np.cos(np.radians(max_tilt)):
            continue
        inl = np.abs((p - a) @ n) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    q = p[best]                                         # 内点最小二乘精修
    c = q.mean(0)
    n = np.linalg.svd(q - c)[2][-1]
    return n, c, np.abs((p - c) @ n) < thr


def estimate_table_frame(pts, valid, mask, hand_boxes, thr=0.005, min_inlier=0.3, ring=60):
    """物体周围一圈 (mask 膨胀 ring px, 去掉膨胀 15 px 以内) 、手框以外、物体顶边以下的有效点拟合桌面
       (物体放在桌上, 它周围就是桌面; 用整张图会拟合到别的平面, 比如笔记本键盘)
       -> R_cam2table, 桌面上一点, 内点比例, 隐含下倾角"""
    keep = valid & dilate(mask, ring) & ~dilate(mask, 15)
    for b in hand_boxes:
        x1, y1, x2, y2 = np.round(b).astype(int)
        keep[max(0, y1):y2, max(0, x1):x2] = False
    keep[:np.where(mask)[0].min()] = False             # 物体顶边以上多半是墙 / 背景
    p = pts[keep]
    n, c, inl = fit_plane(p, thr)
    ratio = inl.mean()
    if ratio < min_inlier:
        raise SystemExit(f"桌面拟合内点只有 {ratio:.0%} (< {min_inlier:.0%}), 换一个参考帧 --ref-frame")
    if n @ (-c) < 0:                                    # 法向朝相机一侧
        n = -n
    zc = np.array([0, 0, 1.0])
    x = zc - (zc @ n) * n; x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(n, x), n])               # 行 = 桌面系各轴在相机系里的方向
    pitch = float(np.degrees(np.arcsin(-zc @ n)))
    return R, c, float(ratio), pitch


def to_table(p_cam, scene):
    return (p_cam - np.array(scene["p_table"])) @ np.array(scene["R_cam2table"]).T


def to_cam(p_tab, scene):
    return p_tab @ np.array(scene["R_cam2table"]) + np.array(scene["p_table"])


def project(p_cam, scene):
    """相机系点 -> 像素 (u, v)"""
    K = np.array(scene["K"]) * [[scene["width"]], [scene["height"]], [1]]
    uv = p_cam @ K.T
    return uv[:, :2] / uv[:, 2:3]


# ================= 3. 点云 =================
def dilate(mask, px):
    return cv2.dilate(mask.astype(np.uint8), np.ones((2 * px + 1, 2 * px + 1), np.uint8)).astype(bool)


def observed_points(pts, valid, rgb, mask, scene):
    """腐蚀 3 px 后的 mask 内的点 (桌面坐标系) 和颜色"""
    core = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & valid
    return to_table(pts[core], scene), rgb[core] / 255.0


def remove_outliers(p, c, k=3.0):
    """到中位点的距离超过 中位距离 + k * MAD 的点删掉"""
    d = np.linalg.norm(p - np.median(p, 0), axis=1)
    med = np.median(d)
    keep = d <= med + k * 1.4826 * np.median(np.abs(d - med))
    return p[keep], c[keep]


def pca_axes(p):
    """俯视 (xy) PCA: 中心, 长轴, 短轴 (都是 2D)"""
    m = p[:, :2].mean(0)
    w, V = np.linalg.eigh(np.cov((p[:, :2] - m).T))
    return m, V[:, 1], V[:, 0]


def chamfer(a, b):
    from scipy.spatial import cKDTree
    return 0.5 * (cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean())


def conflicts(q, scene, depth, margin=0.01):
    """镜像点里落进 "相机看得到的空地" 的比例: 投影到参考帧, 比该像素观测到的深度更靠近相机 (超过 margin)"""
    if depth is None:
        return 0.0
    pc = to_cam(q, scene)
    uv = np.round(project(pc, scene)).astype(int)
    H, W = depth.shape
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (pc[:, 2] > 0)
    d = np.full(len(q), np.nan)
    d[ok] = depth[uv[ok, 1], uv[ok, 0]]
    bad = ok & np.isfinite(d) & (pc[:, 2] < d - margin)
    return bad.mean()


def mirror_complete(p, c, scene=None, depth=None, w_conflict=0.05, span=0.08, step=0.001):
    """对称镜像补全: 对称面过俯视长轴、垂直桌面, 沿短轴搜位置 s, 代价 = Chamfer + w_conflict * 冲突比例"""
    m, u, v = pca_axes(p)
    n3 = np.array([v[0], v[1], 0.0])
    t = (p[:, :2] - m) @ v                              # 观测点在短轴上的坐标
    sub = p[np.random.default_rng(0).choice(len(p), min(len(p), 3000), replace=False)]
    ts = (sub[:, :2] - m) @ v
    best = None
    for s in np.arange(t.min() - span / 2, t.max() + span / 2, step):
        q = sub - 2 * (ts - s)[:, None] * n3
        cost = chamfer(q, sub) + w_conflict * conflicts(q, scene, depth)
        if best is None or cost < best[0]:
            best = (cost, s)
    s = best[1]
    q = p - 2 * (t - s)[:, None] * n3
    info = {"offset": float(s), "conflict": conflicts(q, scene, depth), "axis": u.tolist()}
    return np.r_[p, q], np.r_[c, c], info


def hull_grid(p, step):
    """俯视凸包内的网格点 (xy)"""
    lo, hi = p[:, :2].min(0), p[:, :2].max(0)
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0] + step, step), np.arange(lo[1], hi[1] + step, step), indexing="ij")
    g = np.stack([gx.ravel(), gy.ravel()], 1)
    hull = cv2.convexHull(p[:, :2].astype(np.float32))
    inside = np.array([cv2.pointPolygonTest(hull, (float(x), float(y)), False) >= 0 for x, y in g])
    return g[inside]


def nearest(a, b):
    from scipy.spatial import cKDTree
    return cKDTree(b).query(a)[1]


def add_bottom(p, c, step, z):
    """在桌面高度加一层底面: 俯视凸包内的网格点, 颜色取最近的点"""
    g = hull_grid(p, step)
    q = np.column_stack([g, np.full(len(g), z)])
    return np.r_[p, q], np.r_[c, c[nearest(g, p[:, :2])]]


def fill_column(p, c, step, bottom):
    """竖直补柱子: 俯视凸包内没点的格子补顶面点 (高度/颜色取最近的观测点), 再每个点竖直向下补到桌面"""
    g = hull_grid(p, step)
    cell = set(map(tuple, np.floor(p[:, :2] / step).astype(int)))
    g = np.array([x for x in g if tuple(np.floor(x / step).astype(int)) not in cell]).reshape(-1, 2)
    if len(g):
        k = nearest(g, p[:, :2])
        p, c = np.r_[p, np.column_stack([g, p[k, 2]])], np.r_[c, c[k]]
    n = np.maximum(0, np.floor((p[:, 2] - bottom) / step)).astype(int) + 1
    idx = np.repeat(np.arange(len(p)), n)
    k = np.arange(len(idx)) - np.repeat(np.cumsum(n) - n, n)      # 每个点下面第几个
    q = p[idx].copy()
    q[:, 2] = np.maximum(q[:, 2] - k * step, bottom)
    return q, c[idx]


def voxel_down(p, c, voxel):
    """每个体素保留一个点 (体素内平均)"""
    _, inv = np.unique(np.floor(p / voxel).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.ravel()
    cnt = np.bincount(inv)[:, None]
    avg = lambda x: np.stack([np.bincount(inv, x[:, j]) for j in range(3)], 1) / cnt
    return avg(p), avg(c)


def build_cloud(p, c, voxel=0.007, fill="mirror", scene=None, depth=None):
    """桌面坐标系里的观测点 (桌面 z = 0) -> (物体系点, 颜色, 半径, 包围盒尺寸, 实际体素, 底面中心, 补全信息)"""
    p, c = remove_outliers(p, c)
    above = p[:, 2] > MIN_Z
    p, c = p[above], c[above]
    info = {}
    if fill == "mirror":
        p, c, info = mirror_complete(p, c, scene, depth)
    for _ in range(20):
        r = R_FACTOR * voxel
        if fill == "mirror":
            q, qc = add_bottom(p, c, voxel, r)
            q[:, 2] = np.maximum(q[:, 2], r)            # 最低的小球底面贴桌面
        else:
            q, qc = fill_column(p, c, voxel, r)
        q, qc = voxel_down(q, qc, voxel)
        if len(q) > MAX_N:
            voxel *= 1.1
        elif len(q) < MIN_N and voxel > 0.003:
            voxel /= 1.1
        else:
            break
    if not MIN_N // 2 <= len(q) <= MAX_N:
        raise SystemExit(f"点数 {len(q)} 不在 {MIN_N}~{MAX_N}: 检查 mask.png, 或者调 --voxel")
    # 小球球心在表面上, 外包络会比表面大一个半径: 每个轴把球心线性缩进, 让外包络 = 表面的包围盒 (底面 z = 0)
    s_lo, s_hi = np.r_[p[:, :2].min(0), 0.0], p.max(0)
    q_lo, q_hi = q.min(0), q.max(0)
    q = s_lo + r + (q - q_lo) * (s_hi - s_lo - 2 * r) / np.maximum(q_hi - q_lo, 1e-6)
    lo, hi = q.min(0) - r, q.max(0) + r                 # 包围盒含小球半径
    center = (lo + hi) / 2
    pose = np.array([center[0], center[1], lo[2]])      # 底面中心 (桌面坐标系)
    return q - center, qc, r, hi - lo, voxel, pose, info


def obb_dims(pts, r):
    """沿俯视主轴的 长 / 宽 / 高 (含小球半径)"""
    _, u, v = pca_axes(pts)
    return [float(np.ptp(pts[:, :2] @ u) + 2 * r), float(np.ptp(pts[:, :2] @ v) + 2 * r), float(np.ptp(pts[:, 2]) + 2 * r)]


def save_preview(path, pts, cols, r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (i, j, name) in zip(axs, [(0, 1, "top (x-y)"), (1, 2, "front (y-z)"), (0, 2, "side (x-z)")]):
        o = np.argsort(pts[:, 3 - i - j])        # 远处的先画
        ax.scatter(pts[o, i], pts[o, j], c=cols[o], s=(r * 2 * 72 / 0.02 * 0.5) ** 2 / 4, edgecolors="k", linewidths=0.2)
        ax.set_aspect("equal"); ax.set_title(name)
        ax.set_xlabel("xyz"[i] + " (m)"); ax.set_ylabel("xyz"[j] + " (m)"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def save_asset(out, pts, cols, r, size, voxel, pose, mass, **extra):
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "points.npy"), pts.astype(np.float32))
    np.save(os.path.join(out, "colors.npy"), np.clip(cols, 0, 1).astype(np.float32))
    meta = {"type": "points", "n": len(pts), "radius": round(float(r), 5), "size": np.round(size, 4).tolist(),
            "dims_obb": np.round(obb_dims(pts, r), 4).tolist(), "mass": mass, "voxel": round(float(voxel), 5),
            "pose_table": np.round(pose, 4).tolist(), **extra}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    save_preview(os.path.join(out, "preview.png"), pts, cols, r)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--hoi-json", required=True)
    ap.add_argument("--hoi-repo", required=True, help="frames.py 所在目录")
    ap.add_argument("--name", required=True, help="物体名, 输出到 assets/<name>")
    ap.add_argument("--ref-frame", type=int, default=None, help="参考帧 (物体静止、手离得远); 默认自动挑")
    ap.add_argument("--depth-npz", default=None, help="moge_depth.py 的输出, 用它的 fov_x 保证和手轨迹尺度一致")
    ap.add_argument("--fov-x", type=float, default=None, help="水平视场角 (度); 不给就从 --depth-npz 读或让 MoGe 估")
    ap.add_argument("--fill", default="mirror", choices=["mirror", "column"], help="背面补全: 对称镜像 / 竖直补柱子")
    ap.add_argument("--voxel", type=float, default=0.007, help="体素边长 (m)")
    ap.add_argument("--mass", type=float, default=0.1, help="物体质量 (kg)")
    ap.add_argument("--out-root", default=os.path.join(HERE, "assets"))
    args = ap.parse_args()

    sys.path.insert(0, args.hoi_repo)
    from frames import FrameBoxes
    fb = FrameBoxes(args.hoi_json)
    box0 = prompt_box(fb, pick_frame(fb) or 0)
    i = args.ref_frame if args.ref_frame is not None else choose_ref_frame(fb, box0)
    box = prompt_box(fb, i)
    out = os.path.join(args.out_root, args.name)
    os.makedirs(out, exist_ok=True)
    print(f"参考帧 {i}, SAM2 提示框 {np.round(box).astype(int).tolist()}")

    fov = args.fov_x if args.fov_x else float(np.load(args.depth_npz)["fov_x"]) if args.depth_npz else None
    rgb = read_frame(args.video, i)
    mask = segment(rgb, box)
    cv2.imwrite(os.path.join(out, "mask.png"), mask.astype(np.uint8) * 255)
    pts, valid, K, fov = moge_infer(rgb, fov)
    hands = [b for b in (fb[i].left, fb[i].right, *fb[i].hands) if b is not None]
    R, p0, ratio, pitch = estimate_table_frame(pts, valid, mask, hands)
    scene = {"R_cam2table": R.round(6).tolist(), "p_table": p0.round(5).tolist(), "ref_frame": int(i),
             "fov_x": round(fov, 3), "K": K.round(6).tolist(), "width": rgb.shape[1], "height": rgb.shape[0],
             "pitch_deg": round(pitch, 2), "table_inlier": round(ratio, 3)}
    json.dump(scene, open(os.path.join(out, "scene.json"), "w"), indent=1)
    print(f"桌面: 内点 {ratio:.0%}, 隐含相机下倾角 {pitch:.1f}°, fov_x {fov:.1f}°")
    if not 10 <= pitch <= 30:
        print(f"警告: 下倾角 {pitch:.1f}° 不在 10°~30°, 检查桌面拟合")

    p, c = observed_points(pts, valid, rgb, mask, scene)
    depth = np.where(valid, pts[..., 2], np.nan)
    q, qc, r, size, voxel, pose, info = build_cloud(p, c, args.voxel, args.fill, scene, depth)
    meta = save_asset(out, q, qc, r, size, voxel, pose, args.mass, ref_frame=int(i), fill=args.fill, **info)
    print(f"saved {out}: {meta['n']} 个点, 体素 {voxel * 1000:.1f} mm, 长宽高 {np.round(np.array(meta['dims_obb']) * 100, 1).tolist()} cm, "
          f"底面中心 {meta['pose_table']}" + (f", 对称面冲突 {info['conflict']:.1%}" if info else ""))


if __name__ == "__main__":
    main()
