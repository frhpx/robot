"""
手部 21 个关键点的 3D 轨迹 (桌面坐标系, 和 track_object.py 的物体位姿同一个坐标系), 给 grasp_glue.py 用

  python hand_track.py --video data/mouse.mp4 --object assets/mouse_pts --cache work/mouse_cache.npz --hoi-json mouse.json

输入: track_object.py 的缓存 (MoGe 深度 zs) 和 <object>/track.npz (每帧尺度 scale), scene.json (内参, 桌面系)
输出 <object>/hand.npz:
  kp (N, 21, 3)   MediaPipe 21 点 (0 腕, 4/8/12/16/20 = 拇指/食指/中指/无名指/小指指尖), 桌面系, 米
  valid (N,)      这一帧检测到右手 (无效帧是插值的)
  depth_gap (N,)  手掌深度: MediaPipe 米制手型解出来的 - MoGe 的 (m), 两者一致说明尺度可信

做法:
  1. 选手: HOI json 里离物体 (track.npz 投影) 最近的手框 = 操作物体的手; 框放大 30% 裁出来、放大到 384 px,
     MediaPipe 静态模式检测 (整帧检测时, 手背朝相机、手指弯在物体下面的手常常漏检)
     没给 --hoi-json: 整帧检测, 取离物体最近的手
  2. 每个关键点: 像素 -> 视线; 深度 = 手掌深度 + world_landmarks 给的相对深度 (米制)
     -> 重投影误差为 0, 手型的前后关系保留
  3. 手掌深度: 手掌多边形 (腕 + 指根) 里 MoGe 深度的中值 (乘上 track.npz 的尺度, 和物体同一尺度) + 皮肤到关节约 1 cm
     MoGe 没有有效深度时, 退回 "手型不变只解平移" 的 MediaPipe 米制深度
  4. 相机系 -> 桌面系; 漏检帧插值; 时间平滑
"""
import argparse, json, os, sys
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from object_points import to_table, to_cam, project
from track_object import read_video

PALM = [0, 5, 9, 13, 17]       # 腕 + 4 个指根: 比指尖稳, 深度用它们
TIPS = [4, 8, 12, 16, 20]
SKIN = 0.01                    # 关节中心在可见皮肤后面约 1 cm


def rays(uv, scene):
    """像素 -> 相机系单位视线方向"""
    K = np.array(scene["K"]) * [[scene["width"]], [scene["height"]], [1]]
    d = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(K).T
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def solve_translation(shape, d):
    """t 使 t + shape_i 落在视线 d_i 上: sum (I - d d^T)(t + shape_i) = 0"""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for s, di in zip(shape, d):
        P = np.eye(3) - np.outer(di, di)
        A += P; b -= P @ s
    return np.linalg.solve(A, b)


def mp_hands(rgb, hands):
    """一张图里所有手: [(像素 (21, 2), 手型 (21, 3))]"""
    H, W = rgb.shape[:2]
    res = hands.process(rgb)
    return [(np.array([[p.x * W, p.y * H] for p in lm.landmark]), np.array([[p.x, p.y, p.z] for p in wl.landmark]))
            for lm, wl in zip(res.multi_hand_landmarks or [], res.multi_hand_world_landmarks or [])]


def detect(rgbs, boxes, obj_uv, crop_px=384):
    """每帧操作物体的那只手 (像素, 手型) 或 None. boxes[i]: 这一帧的手框列表 (没给 HOI json 时为 None)"""
    import mediapipe as mp
    out = []
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, model_complexity=1,
                                  min_detection_confidence=0.2) as crop_det,          mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=1,
                                  min_detection_confidence=0.3, min_tracking_confidence=0.3) as full_det:
        for i, rgb in enumerate(rgbs):
            H, W = rgb.shape[:2]
            found = None
            if boxes is not None and boxes[i]:
                b = min(boxes[i], key=lambda b: np.hypot((b[0] + b[2]) / 2 - obj_uv[i, 0], (b[1] + b[3]) / 2 - obj_uv[i, 1]))
                cx, cy, half = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 0.65 * max(b[2] - b[0], b[3] - b[1])
                x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
                x2, y2 = int(min(W, cx + half)), int(min(H, cy + half))
                k = crop_px / max(x2 - x1, y2 - y1)
                crop = cv2.resize(rgb[y1:y2, x1:x2], (int((x2 - x1) * k), int((y2 - y1) * k)))
                hs = mp_hands(crop, crop_det)
                if hs:
                    uv, shape = hs[0]
                    found = (uv * [(x2 - x1) / crop.shape[1], (y2 - y1) / crop.shape[0]] + [x1, y1], shape)
            if found is None:
                hs = mp_hands(rgb, full_det)
                if hs and boxes is None:
                    found = min(hs, key=lambda h: np.linalg.norm(h[0].mean(0) - obj_uv[i]))
            out.append(found)
    return out


def palm_depth(z, uv):
    """手掌多边形 (腕 + 指根) 里有效深度的中值"""
    m = np.zeros(z.shape, np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(np.round(uv[[0, 1, 5, 9, 13, 17]]).astype(np.int32)), 1)
    p = z[(m > 0) & np.isfinite(z)]
    return float(np.median(p)) if len(p) >= 20 else np.nan


def hoi_hand_boxes(path, n):
    raw = json.load(open(path, encoding="utf-8"))
    boxes = [[] for _ in range(n)]
    for fr in raw["frames"]:
        if fr["frame_idx"] < n:
            boxes[fr["frame_idx"]] = [d["box"] for d in fr["detections"] if d["class_name"] == "hand"]
    return boxes


def smooth(x, win=5):
    """(N, ...) 沿时间: 5 帧中值 + win 帧滑动平均"""
    sh = x.shape
    x = x.reshape(len(x), -1)
    p = np.pad(x, ((2, 2), (0, 0)), mode="edge")
    x = np.median(np.stack([p[i:i + len(x)] for i in range(5)]), 0)
    k = np.ones(win) / win
    x = np.stack([np.convolve(np.pad(x[:, c], win // 2, mode="edge"), k, "valid") for c in range(x.shape[1])], 1)
    return x.reshape(sh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--object", required=True, help="有 scene.json 和 track.npz (track_object.py 的输出)")
    ap.add_argument("--cache", required=True, help="track_object.py --cache 的文件 (MoGe 深度)")
    ap.add_argument("--hoi-json", default=None, help="HOI-DETR json: 用手框裁图再检测 (推荐)")
    ap.add_argument("--out", default=None, help="默认 <object>/hand.npz")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    scene = json.load(open(os.path.join(args.object, "scene.json")))
    tr = np.load(os.path.join(args.object, "track.npz"))
    zs = np.load(args.cache, allow_pickle=True)["zs"]
    rgbs, fps = read_video(args.video)
    n = len(rgbs)
    obj_uv = project(to_cam(tr["t"], scene), scene)     # 物体中心的像素位置 (选手用)

    boxes = hoi_hand_boxes(args.hoi_json, n) if args.hoi_json else None
    dets = detect(rgbs, boxes, obj_uv)
    kp = np.full((n, 21, 3), np.nan)
    uv_all = np.full((n, 21, 2), np.nan)
    gap = np.full(n, np.nan)
    for i, det in enumerate(dets):
        if det is None:
            continue
        uv, shape = det
        d = rays(uv, scene)
        z_mp = (shape + solve_translation(shape, d))[:, 2]            # MediaPipe 米制手型给的深度
        zm = palm_depth(zs[i] * tr["scale"][i], uv)
        rel = shape[:, 2] - np.median(shape[PALM, 2])                   # 各点相对手掌的深度
        if np.isfinite(zm):
            gap[i] = np.median(z_mp[PALM]) - (zm + SKIN)
            z = zm + SKIN + rel
        else:
            z = np.median(z_mp[PALM]) + rel
        pc = d * (z / d[:, 2])[:, None]                                 # 沿视线到深度 z
        kp[i] = to_table(pc, scene)
        uv_all[i] = uv

    valid = np.isfinite(kp[:, 0, 0])
    if valid.sum() < 2:
        raise SystemExit("几乎没检测到右手")
    idx = np.where(valid)[0]
    flat = kp.reshape(n, -1)
    for c in range(flat.shape[1]):
        flat[:, c] = np.interp(np.arange(n), idx, flat[idx, c])
    kp = smooth(flat.reshape(n, 21, 3))

    path = args.out or os.path.join(args.object, "hand.npz")
    np.savez(path, kp=kp, valid=valid, depth_gap=gap, uv=uv_all, fps=fps)
    if not args.no_video:
        import imageio
        frames = []
        bones = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
                 (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
        for i in range(n):
            img = cv2.cvtColor(rgbs[i], cv2.COLOR_RGB2BGR)
            q = np.round(project(to_cam(kp[i], scene), scene)).astype(int)
            for a, b in bones:
                cv2.line(img, tuple(q[a]), tuple(q[b]), (0, 255, 255) if valid[i] else (128, 128, 128), 1)
            for k in TIPS:
                cv2.circle(img, tuple(q[k]), 3, (0, 0, 255), -1)
            cv2.putText(img, str(i), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        imageio.mimsave(os.path.join(args.object, "hand.mp4"), frames, fps=fps, macro_block_size=1)
    g = gap[np.isfinite(gap)]
    print(f"手检测到 {valid.sum()}/{n} 帧; "
          f"手掌深度 MediaPipe - MoGe: median {np.median(g) * 100:.1f} cm, |p90| {np.percentile(np.abs(g), 90) * 100:.1f} cm")
    print("saved", path)


if __name__ == "__main__":
    main()
