"""
验收: 仿真初始时刻的物体点云投影回原视频, 和 SAM2 mask 比 IoU

  python check_alignment.py --object assets/mouse_pts --video data/mouse.mp4 --hoi-json data/hoi/mouse.json \
      --hoi-repo ~/code/handbench/hoi --menagerie ~/mujoco_menagerie
  python check_alignment.py ... --frame2 350      # 另一帧 (默认自动挑: 离参考帧 >= 100 帧、没有手的帧)

做法: build_model 建场景 -> mj_forward -> 物体各小球的世界坐标 -> 逆变换回相机系 (scene.json) -> 用 MoGe 内参投影,
每个点画半径 3 px 的实心圆 -> 和 SAM2 mask 算 IoU.
另外算 "小球实际投影" 的 IoU: 每个点按小球半径和它的深度画圆 (r_px = f * r / z), 这才是仿真里物体在画面上占的区域;
体素 7 mm、距离 0.37 m 时点间距约 10 px, 3 px 的圆盖不满, 所以两个都打印.
另一帧重新跑 SAM2 (提示框 = 参考帧 mask 的外框), 相机不动时投影不变.
输出 <object>/overlay.png: 左参考帧, 右另一帧; 绿 = mask 轮廓, 红 = 投影点
"""
import argparse, json, os, sys
import numpy as np
import cv2
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hoi_to_mujoco as H
from object_asset import read_frame, segment


def projected_mask(uv, shape, radius=3):
    """radius: 统一的像素半径, 或者每个点一个半径"""
    m = np.zeros(shape, np.uint8)
    rad = np.broadcast_to(radius, len(uv))
    for (u, v), r in zip(np.round(uv).astype(int), rad):
        cv2.circle(m, (int(u), int(v)), max(1, int(round(r))), 1, -1)
    return m.astype(bool)


def iou(a, b):
    return (a & b).sum() / max(1, (a | b).sum())


def overlay(rgb, mask, uv, label):
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, (0, 255, 0), 2)
    for u, v in np.round(uv).astype(int):
        cv2.circle(img, (int(u), int(v)), 3, (0, 0, 255), -1)
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--hoi-json", required=True)
    ap.add_argument("--hoi-repo", required=True)
    ap.add_argument("--menagerie", default=os.path.expanduser("~/mujoco_menagerie"))
    ap.add_argument("--frame2", type=int, default=None, help="另一帧 (和参考帧相差 >= 100 帧的静止画面)")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.object, "meta.json")))
    scene = json.load(open(os.path.join(args.object, "scene.json")))
    H.OBJ_POS = np.array([*H.OBJ_XY, H.TABLE_TOP + meta["size"][2] / 2])
    m = H.build_model(args.menagerie, args.object)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    ob = m.body("obj").id
    world = d.geom_xpos[m.geom_bodyid == ob]
    K = np.array(scene["K"]) * [[scene["width"]], [scene["height"]], [1]]
    pc = H.world_to_cam(world, scene, meta)
    uv = pc @ K.T
    uv = uv[:, :2] / uv[:, 2:3]

    sys.path.insert(0, args.hoi_repo)
    from frames import FrameBoxes
    fb = FrameBoxes(args.hoi_json)
    ref = scene["ref_frame"]
    f2 = args.frame2
    if f2 is None:
        free = [f.idx for f in fb if abs(f.idx - ref) >= 100 and not f.hands and f.left is None and f.right is None]
        if not free:
            raise SystemExit("找不到和参考帧相差 >= 100 帧、没有手的帧, 用 --frame2 指定")
        f2 = free[len(free) // 2]

    mask_ref = cv2.imread(os.path.join(args.object, "mask.png"), 0) > 0
    ys, xs = np.where(mask_ref)
    pad = 15
    rgb2 = read_frame(args.video, f2)
    mask2 = segment(rgb2, [xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad])
    proj = projected_mask(uv, mask_ref.shape)
    proj_sph = projected_mask(uv, mask_ref.shape, K[0, 0] * meta["radius"] / pc[:, 2])
    res = {}
    imgs = []
    for name, f, mk, rgb in [("ref", ref, mask_ref, read_frame(args.video, ref)), ("frame2", f2, mask2, rgb2)]:
        res[name] = (f, iou(proj, mk), iou(proj_sph, mk))
        imgs.append(overlay(rgb, mk, uv, f"frame {f}  IoU {res[name][1]:.2f} / {res[name][2]:.2f}"))
    cv2.imwrite(os.path.join(args.object, "overlay.png"), np.hstack(imgs))
    for name, (f, v, vs) in res.items():
        need = 0.80 if name == "ref" else 0.70
        print(f"{name}: frame {f}, IoU (3 px) {v:.3f} ({'OK' if v >= need else 'FAIL'} >= {need}), "
              f"IoU (小球投影) {vs:.3f} ({'OK' if vs >= need else 'FAIL'})")
    print("saved", os.path.join(args.object, "overlay.png"))


if __name__ == "__main__":
    main()
