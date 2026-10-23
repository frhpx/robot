"""
CPU 版 HOI 检测 (没有 GPU / HOI-DETR 时用): 输出和 HOI-DETR 相同格式的 json, hoi_to_mujoco.py 直接可用

  手:   MediaPipe Hands (需要 mediapipe==0.10.14, 自带模型; Python 3.11), 左右手用它自带的判断,
        写进每个手框的 "side" 字段 (hoi_to_mujoco.py 优先用它)
  物体: YOLO (COCO 类别, 例如 mouse / cup / bottle); YOLO 漏检 (常因被手握住) 时用 CSRT 跟踪器补
        (需要 opencv-contrib-python)
  接触: 手框和物体框的交集 / 物体框面积 > CONTACT_OVERLAP -> 写一条 hf 链接

  python hoi_cpu.py --video 4.mp4 --object-class mouse --out 4.json
"""
import argparse, json
import numpy as np
import cv2

CONTACT_OVERLAP = 0.25


def overlap(h, o):
    w = min(h[2], o[2]) - max(h[0], o[0])
    hh = min(h[3], o[3]) - max(h[1], o[1])
    return max(0, w) * max(0, hh) / ((o[2] - o[0]) * (o[3] - o[1]) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--object-class", default="mouse", help="COCO 类别名")
    ap.add_argument("--out", required=True)
    ap.add_argument("--yolo", default="yolo11n.pt")
    args = ap.parse_args()

    import mediapipe as mp
    from ultralytics import YOLO
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,      # 跟踪模式召回最高
                                     min_detection_confidence=0.3, min_tracking_confidence=0.3)
    yolo = YOLO(args.yolo)
    cls = [k for k, v in yolo.names.items() if v == args.object_class][0]

    cap = cv2.VideoCapture(args.video)
    fps, W, H = cap.get(cv2.CAP_PROP_FPS), int(cap.get(3)), int(cap.get(4))
    hand_boxes, obj, tracked = [], [], []
    tracker = None
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        hb = []
        for lm, hd in zip(res.multi_hand_landmarks or [], res.multi_handedness or []):
            xy = np.array([[p.x * W, p.y * H] for p in lm.landmark])
            pad = 0.1 * (xy.max(0) - xy.min(0))
            # MediaPipe 假设输入是镜像 (自拍) 画面; 普通视频里它说 "Left" 的是真实的右手
            side = "Right" if hd.classification[0].label == "Left" else "Left"
            hb.append(([*(xy.min(0) - pad).tolist(), *(xy.max(0) + pad).tolist()], side))
        hand_boxes.append(hb)
        r = yolo(bgr, classes=[cls], conf=0.2, verbose=False)[0]
        if len(r.boxes):                                   # YOLO 检到: 用它, 并重置跟踪器
            b = r.boxes.xyxy[int(r.boxes.conf.argmax())].tolist()
            tracker = cv2.TrackerCSRT_create()
            tracker.init(bgr, (int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])))
            obj.append(b); tracked.append(False)
        elif tracker is not None:                          # 没检到: 跟踪器补
            ok_t, (x, y, w, h) = tracker.update(bgr)
            obj.append([x, y, x + w, y + h] if ok_t else None); tracked.append(ok_t)
        else:
            obj.append(None); tracked.append(False)

    # 物体漏检 (常被手挡住): 线性插值, 首尾外推为最近一次检测
    n = len(obj)
    ob = np.array([o if o is not None else [np.nan] * 4 for o in obj])
    ok = ~np.isnan(ob[:, 0])
    print(f"{n} 帧, 物体有框 {ok.sum()} 帧 (其中跟踪器补 {sum(tracked)} 帧), "
          f"有右手 {sum(1 for h in hand_boxes if any(s == 'Right' for _, s in h))} 帧")
    t = np.arange(n)
    for c in range(4):
        ob[:, c] = np.interp(t, t[ok], ob[ok, c])

    frames = []
    for i in range(n):
        dets = [{"box": b, "score": 0.9, "class_id": 0, "class_name": "hand", "side": sd} for b, sd in hand_boxes[i]]
        dets.append({"box": ob[i].tolist(), "score": 0.9 if ok[i] else 0.5, "class_id": 1, "class_name": "firstobject"})
        hf = [{"a": k, "b": len(dets) - 1, "prob": 0.9}
              for k, (b, _) in enumerate(hand_boxes[i]) if overlap(b, ob[i]) > CONTACT_OVERLAP]
        frames.append({"frame_idx": i, "processed": True, "detections": dets, "hf": hf, "fs": []})
    meta = {"type": "video", "video_path": args.video, "fps": fps, "width": W, "height": H, "num_frames": n,
            "score_thr": 0.5, "nms_iou": 0.5, "frame_stride": 1,
            "class_names": ["hand", "firstobject", "secondobject"], "frames": frames,
            "source": f"hoi_cpu.py (MediaPipe + YOLO {args.object_class})"}
    json.dump(meta, open(args.out, "w"))
    print("saved", args.out)


if __name__ == "__main__":
    main()
