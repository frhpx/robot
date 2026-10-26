"""
从视频里的一帧生成点云物体资产 (小球拼成的刚体), 不生成 mesh; hoi_to_mujoco.py --object 直接用

  python object_points.py --video data/mouse.mp4 --hoi-json data/hoi/mouse.json --hoi-repo ~/code/handbench/hoi \
      --name mouse_pts --cam-pitch 15
  python object_points.py ... --frame 120          # 指定用哪一帧 (默认自动挑物体最清楚的一帧)
  python object_points.py ... --voxel 0.007        # 体素边长 (m), 点太多会自动加大

--cam-pitch 必须和之后运行 hoi_to_mujoco.py 时一致 (写进 meta.json, 不一致时 hoi_to_mujoco.py 会警告)

输出 assets/<name>/:
  points.npy   (N, 3) 物体坐标系, 米, 原点在包围盒中心 (同 mesh 资产的约定)
  colors.npy   (N, 3) 每个点的颜色 0~1
  meta.json    type="points", n, radius (小球半径), size (包围盒尺寸 m, 含小球半径), mass, frame, cam_pitch,
               voxel (实际体素), align_yaw (对齐长轴时绕 z 转的角度, 度)
  mask.png     分割结果, preview.png 俯视/正视/侧视图 (用来检查)

流程:
  1. 挑帧 (同 object_asset.py)
  2. SAM2 分割, mask 腐蚀 3 px 去掉边缘深度跳变的飞点
  3. MoGe-2 米制点图, 取 mask 内的点转到机器人系
  4. 去离群点: 到中位点的距离超过 中位距离 + 3 MAD
  5. 桌面高度: mask 外围一圈 (膨胀 25 px 减去膨胀 5 px) 的点的 z 中位数
  6. 补成实心: 单帧只看得到朝向相机的一面. 先绕 z 转到俯视长轴 = y (同 mesh 资产, size 就是物体自身的长宽高);
     俯视凸包内没点的格子补顶面点 (侧拍时只看到侧壁, 俯视是一条弧); 再每个点按 voxel 间距竖直向下补到桌面
  7. 体素降采样 (每个体素取平均); 超过 MAX_N 个点就加大体素
  8. 颜色: 像素 RGB, 补出来的点继承上方那个点的颜色
  9. 平移到包围盒中心; 小球半径 r = 0.6 voxel
"""
import argparse, json, os, sys
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from object_asset import pick_frame, read_frame, segment
from hoi_to_mujoco import cam_to_robot_R

MAX_N, MIN_N = 400, 60
R_FACTOR = 0.6              # 小球半径 / 体素边长


def moge_points(rgb, model_id="Ruicheng/moge-2-vitl-normal"):
    """MoGe-2: 一帧的米制点图 (H, W, 3, 相机系) + 有效掩码"""
    import torch
    from moge.model.v2 import MoGeModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoGeModel.from_pretrained(model_id).to(dev).eval()
    img = torch.tensor(rgb / 255, dtype=torch.float32, device=dev).permute(2, 0, 1)
    with torch.no_grad():
        out = model.infer(img, use_fp16=(dev == "cuda"))
    pts, valid = out["points"].cpu().numpy(), out["mask"].cpu().numpy().astype(bool)
    del model; torch.cuda.empty_cache()
    return pts, valid


def dilate(mask, px):
    return cv2.dilate(mask.astype(np.uint8), np.ones((2 * px + 1, 2 * px + 1), np.uint8)).astype(bool)


def observed_points(pts, valid, rgb, mask, pitch):
    """步骤 2 后半 + 3 + 5: 腐蚀后的 mask 内的点 (机器人系) 和颜色, 以及桌面高度"""
    R = cam_to_robot_R(pitch)
    core = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & valid
    ring = dilate(mask, 25) & ~dilate(mask, 5) & valid
    if ring.sum() < 50:
        raise SystemExit("mask 外围有效深度点太少, 估不出桌面高度")
    table_z = float(np.median((pts[ring] @ R.T)[:, 2]))
    return pts[core] @ R.T, rgb[core] / 255.0, table_z


def remove_outliers(p, c, k=3.0):
    """到中位点的距离超过 中位距离 + k * MAD 的点删掉"""
    d = np.linalg.norm(p - np.median(p, 0), axis=1)
    med = np.median(d)
    keep = d <= med + k * 1.4826 * np.median(np.abs(d - med))
    return p[keep], c[keep]


def align_long_axis(p):
    """绕竖直轴旋转, 让俯视图的主方向 (PCA) 对齐 y 轴 (同 mesh 资产); 返回旋转后的点和转角 (rad)"""
    xy = p[:, :2] - p[:, :2].mean(0)
    v = np.linalg.eigh(np.cov(xy.T))[1][:, -1]                   # 最长的方向
    yaw = np.pi / 2 - np.arctan2(v[1], v[0])
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2                  # 方向不分正反, 取转角最小的那个
    c, s = np.cos(yaw), np.sin(yaw)
    return p @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T, yaw


def fill_footprint(p, c, step):
    """补俯视图的空洞: 侧拍时只看到朝相机的侧壁 (俯视是一条弧), 把俯视凸包内没有点的格子补上顶面点,
       高度和颜色取最近的有点的格子 (该格的最高点)"""
    cell = np.floor(p[:, :2] / step).astype(np.int64)
    keys, inv = np.unique(cell, axis=0, return_inverse=True)
    inv = inv.ravel()
    top = np.full(len(keys), -np.inf); np.maximum.at(top, inv, p[:, 2])
    top_i = np.zeros(len(keys), int)
    top_i[inv[p[:, 2] == top[inv]]] = np.nonzero(p[:, 2] == top[inv])[0]   # 每格最高点的下标
    hull = cv2.convexHull(keys.astype(np.float32))
    lo, hi = keys.min(0), keys.max(0)
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel()], 1)
    have = set(map(tuple, keys))
    new = np.array([g for g in grid if tuple(g) not in have
                    and cv2.pointPolygonTest(hull, (float(g[0]), float(g[1])), False) >= 0]).reshape(-1, 2)
    if len(new) == 0:
        return p, c
    near = np.argmin(((new[:, None, :] - keys[None]) ** 2).sum(-1), 1)
    q = np.column_stack([(new + 0.5) * step, top[near]])
    return np.r_[p, q], np.r_[c, c[top_i[near]]]


def fill_down(p, c, table_z, step, bottom):
    """每个点竖直向下按 step 补点, 补到 table_z + bottom (小球底面贴桌面); 颜色继承上方的点"""
    n = np.maximum(0, np.floor((p[:, 2] - table_z - bottom) / step)).astype(int) + 1
    idx = np.repeat(np.arange(len(p)), n)
    k = np.arange(len(idx)) - np.repeat(np.cumsum(n) - n, n)      # 每个点下面第几个
    q = p[idx].copy()
    q[:, 2] = np.maximum(q[:, 2] - k * step, table_z + bottom)
    return q, c[idx]


def voxel_down(p, c, voxel):
    """每个体素保留一个点 (体素内平均)"""
    _, inv = np.unique(np.floor(p / voxel).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.ravel()
    cnt = np.bincount(inv)[:, None]
    avg = lambda x: np.stack([np.bincount(inv, x[:, j]) for j in range(3)], 1) / cnt
    return avg(p), avg(c)


def build_cloud(p, c, table_z, voxel=0.007):
    """步骤 4 及之后: 机器人系观测点 + 颜色 + 桌面高度 -> (物体系点, 颜色, 半径, 包围盒尺寸, 实际体素, 对齐转角)"""
    p, c = remove_outliers(p, c)
    above = p[:, 2] > table_z                # 低于桌面的点 (阴影 / 桌面误分割) 不要
    p, c = p[above], c[above]
    p, yaw = align_long_axis(p)
    while True:
        r = R_FACTOR * voxel
        q, qc = voxel_down(*fill_down(*fill_footprint(p, c, voxel), table_z, voxel, r), voxel)
        if len(q) <= MAX_N:
            break
        voxel *= 1.15
    if len(q) < MIN_N:
        raise SystemExit(f"点太少 ({len(q)} < {MIN_N}): 检查 mask.png, 或者减小 --voxel")
    lo, hi = q.min(0) - r, q.max(0) + r     # 包围盒含小球半径: 最低的小球底面正好贴桌面
    return q - (lo + hi) / 2, qc, r, hi - lo, voxel, yaw


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


def save_asset(out, pts, cols, r, size, mass, frame, pitch_deg, voxel, yaw):
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "points.npy"), pts.astype(np.float32))
    np.save(os.path.join(out, "colors.npy"), np.clip(cols, 0, 1).astype(np.float32))
    meta = {"type": "points", "n": len(pts), "radius": round(float(r), 5), "size": np.round(size, 4).tolist(),
            "mass": mass, "frame": frame, "cam_pitch": pitch_deg, "voxel": round(float(voxel), 5),
            "align_yaw": round(float(np.degrees(yaw)), 1)}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    save_preview(os.path.join(out, "preview.png"), pts, cols, r)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--hoi-json", required=True)
    ap.add_argument("--hoi-repo", required=True, help="frames.py 所在目录")
    ap.add_argument("--name", required=True, help="物体名, 输出到 assets/<name>")
    ap.add_argument("--cam-pitch", type=float, default=50, help="相机向下倾斜角 (度), 必须和 hoi_to_mujoco.py 一致")
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--voxel", type=float, default=0.007, help="体素边长 (m)")
    ap.add_argument("--mass", type=float, default=0.1, help="物体质量 (kg)")
    ap.add_argument("--out-root", default=os.path.join(HERE, "assets"))
    args = ap.parse_args()

    sys.path.insert(0, args.hoi_repo)
    from frames import FrameBoxes
    fb = FrameBoxes(args.hoi_json)
    i = args.frame if args.frame is not None else pick_frame(fb)
    if i is None or fb[i].obj is None:
        raise SystemExit("这一帧没有物体框")
    out = os.path.join(args.out_root, args.name)
    os.makedirs(out, exist_ok=True)
    print(f"用第 {i} 帧, 物体框 {np.round(fb[i].obj).astype(int).tolist()}")

    rgb = read_frame(args.video, i)
    mask = segment(rgb, fb[i].obj)
    cv2.imwrite(os.path.join(out, "mask.png"), mask.astype(np.uint8) * 255)
    pts, valid = moge_points(rgb)
    p, c, table_z = observed_points(pts, valid, rgb, mask, np.deg2rad(args.cam_pitch))
    print(f"mask 内有效点 {len(p)}, 桌面高度 z = {table_z:.3f} m (相机系原点)")
    q, qc, r, size, voxel, yaw = build_cloud(p, c, table_z, args.voxel)
    meta = save_asset(out, q, qc, r, size, args.mass, int(i), args.cam_pitch, voxel, yaw)
    print(f"saved {out}: {meta['n']} 个点, 体素 {voxel * 1000:.1f} mm, 尺寸 {np.round(size * 100, 1).tolist()} cm")


if __name__ == "__main__":
    main()
