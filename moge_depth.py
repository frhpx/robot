"""
MoGe-2 单目深度 -> 每帧右手 / 物体在相机坐标系里的米制 3D 点, 给 hoi_to_mujoco.py --depth-npz 用

  pip install git+https://github.com/microsoft/MoGe.git
  python moge_depth.py --video data/1.mp4 --hoi-json data/1.json --hoi-repo ~/HOI-pipeline-for-hand-detection --out moge.npz

做法:
  1. 第一帧估一次视场角 (fov), 之后固定, 保证各帧尺度一致
  2. 每帧 MoGe 输出点图 points (H, W, 3), OpenCV 相机系: x右 y下 z前, 单位米
  3. 取右手框中间 50% 区域里有效点的中值 = 手的 3D 位置 (中值去掉框里的背景点); 物体同理
"""
import argparse, sys
import numpy as np
import cv2
import torch


def center_median(points, mask, box, keep=0.5):
    """box (xyxy) 中间 keep 比例区域内有效点的中值, 没有有效点返回 None"""
    if box is None:
        return None
    H, W = mask.shape
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hw, hh = (box[2] - box[0]) * keep / 2, (box[3] - box[1]) * keep / 2
    x1, x2 = int(max(0, cx - hw)), int(min(W, cx + hw))
    y1, y2 = int(max(0, cy - hh)), int(min(H, cy + hh))
    p = points[y1:y2, x1:x2][mask[y1:y2, x1:x2]]
    return np.median(p, 0) if len(p) > 20 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--hoi-json", required=True)
    ap.add_argument("--hoi-repo", required=True, help="frames.py 所在目录")
    ap.add_argument("--out", default="moge.npz")
    ap.add_argument("--model", default="Ruicheng/moge-2-vitl-normal", help="显存不够换 moge-2-vits-normal")
    ap.add_argument("--fov-x", type=float, default=None, help="已知水平视场角 (度) 就填, 否则第一帧估")
    ap.add_argument("--stride", type=int, default=1, help="每 N 帧算一次, 中间帧由 hoi_to_mujoco 插值")
    args = ap.parse_args()

    sys.path.insert(0, args.hoi_repo)
    from frames import FrameBoxes
    from moge.model.v2 import MoGeModel

    fb = FrameBoxes(args.hoi_json)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoGeModel.from_pretrained(args.model).to(dev).eval()

    cap = cv2.VideoCapture(args.video)
    fov_x = args.fov_x
    rec = {k: [] for k in ("frame", "hand_xyz", "obj_xyz", "valid", "valid_obj")}
    for i in range(len(fb)):
        ok, bgr = cap.read()
        if not ok:
            break
        if i % args.stride:
            continue
        img = torch.tensor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255, dtype=torch.float32, device=dev).permute(2, 0, 1)
        with torch.no_grad():
            out = model.infer(img, fov_x=fov_x, use_fp16=(dev == "cuda"))
        if fov_x is None:                                    # 第一帧估出 fov, 之后固定
            fx = out["intrinsics"][0, 0].item()              # 归一化焦距 (除以图像宽)
            fov_x = float(np.degrees(2 * np.arctan(0.5 / fx)))
            print(f"fov_x = {fov_x:.1f}°")
        pts, mask = out["points"].cpu().numpy(), out["mask"].cpu().numpy().astype(bool)
        h, o = center_median(pts, mask, fb[i].right), center_median(pts, mask, fb[i].obj)
        rec["frame"].append(i)
        rec["hand_xyz"].append(h if h is not None else np.full(3, np.nan))
        rec["obj_xyz"].append(o if o is not None else np.full(3, np.nan))
        rec["valid"].append(h is not None)
        rec["valid_obj"].append(o is not None)
        if i % 50 == 0:
            print(f"frame {i}/{len(fb)}  hand {None if h is None else np.round(h, 3)}")

    np.savez(args.out, fov_x=fov_x, **{k: np.array(v) for k, v in rec.items()})
    print("saved", args.out, f"(有效手部帧 {sum(rec['valid'])}/{len(rec['valid'])})")


if __name__ == "__main__":
    main()
