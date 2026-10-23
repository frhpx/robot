"""
从视频里的一帧生成物体资产 (完整 3D 形状 + 真实尺寸), 生成一次, 之后 hoi_to_mujoco.py --object 反复使用

  python object_asset.py --video data/1.mp4 --hoi-json data/1.json --hoi-repo ~/HOI-pipeline-for-hand-detection --name mug
  python object_asset.py ... --frame 120          # 指定用哪一帧 (默认自动挑物体最清楚的一帧)
  python object_asset.py ... --height 0.10        # 已知真实高度 (m) 就填, 否则用 MoGe 量

输出 assets/<name>/:
  mesh.obj         米制, z 朝上, 原点在包围盒中心 (显示用)
  collision_*.obj  CoACD 凸分解的各块 (物理碰撞用)
  meta.json        size (包围盒尺寸 m), rgba, n_collision, density (按 --mass 算), frame, height_source
  crop.png    送进 Hunyuan3D 的抠图, mask.png 分割结果 (用来检查)

流程:
  1. 挑帧: 有物体框的帧里, 选 "物体框面积 x 没被手挡住的比例" 最大的一帧
  2. SAM2: 用 HOI 的物体框做提示, 分割出物体 mask
  3. Hunyuan3D-2mini: 抠图 -> 完整 3D 形状 (约 6 GB 显存; 不做纹理, 纹理要 16 GB)
  4. MoGe-2: 这一帧的米制点图, 取 mask 内的点转到机器人系, 量物体高度
  5. 形状 y 朝上 -> z 朝上, 居中, 按高度缩放到米制
  6. CoACD 凸分解 -> 碰撞体; 密度 = 质量 / 各凸块体积之和
"""
import argparse, json, os, sys
import numpy as np
import cv2
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))


def covered(box, hands):
    """box 被手框挡住的面积比例"""
    a = (box[2] - box[0]) * (box[3] - box[1])
    c = 0.0
    for h in hands:
        w, hh = min(box[2], h[2]) - max(box[0], h[0]), min(box[3], h[3]) - max(box[1], h[1])
        c += max(0, w) * max(0, hh)
    return min(1.0, c / a)


def pick_frame(fb):
    best, score = None, -1
    for f in fb:
        if f.obj is None:
            continue
        b = f.obj
        s = (b[2] - b[0]) * (b[3] - b[1]) * (1 - covered(b, [h for h in (f.left, f.right) if h is not None])) ** 2
        if s > score:
            best, score = f.idx, s
    return best


def read_frame(video, i):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, bgr = cap.read()
    if not ok:
        raise SystemExit(f"读不到第 {i} 帧")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def segment(rgb, box, model_id="facebook/sam2.1-hiera-small"):
    """SAM2 框提示分割, 只保留最大连通域"""
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    pred = SAM2ImagePredictor.from_pretrained(model_id)
    with torch.inference_mode():
        pred.set_image(rgb)
        masks, _, _ = pred.predict(box=np.array(box), multimask_output=False)
    m = masks[0].astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        m = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    del pred; torch.cuda.empty_cache()
    return m.astype(bool)


def rgba_crop(rgb, mask, pad=0.15):
    """mask 作 alpha 通道, 裁成带边距的正方形"""
    ys, xs = np.where(mask)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    r = (1 + pad) * max(xs.max() - xs.min(), ys.max() - ys.min()) / 2
    rgba = np.dstack([rgb, mask.astype(np.uint8) * 255])
    rgba = cv2.copyMakeBorder(rgba, int(r), int(r), int(r), int(r), cv2.BORDER_CONSTANT, value=0)
    cx, cy = cx + int(r), cy + int(r)
    return rgba[int(cy - r):int(cy + r), int(cx - r):int(cx + r)]


def generate_shape(rgba):
    """Hunyuan3D-2mini: RGBA 抠图 -> trimesh (只有形状)"""
    import torch
    from PIL import Image
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover, FaceReducer
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mini", subfolder="hunyuan3d-dit-v2-mini", variant="fp16")
    mesh = pipe(image=Image.fromarray(rgba, "RGBA"), num_inference_steps=50, octree_resolution=380,
                num_chunks=20000, generator=torch.manual_seed(0), output_type="trimesh")[0]
    del pipe; torch.cuda.empty_cache()
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    return FaceReducer()(mesh, max_facenum=20000)        # MuJoCo 里显示用, 2 万面足够


def measure_height(rgb, mask):
    """MoGe-2: mask 内的米制点转到机器人系, 取 z 方向 3%~97% 分位的跨度作高度"""
    import torch
    sys.path.insert(0, HERE)
    from hoi_to_mujoco import cam_to_robot_R
    from moge.model.v2 import MoGeModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(dev).eval()
    img = torch.tensor(rgb / 255, dtype=torch.float32, device=dev).permute(2, 0, 1)
    with torch.no_grad():
        out = model.infer(img, use_fp16=(dev == "cuda"))
    pts, valid = out["points"].cpu().numpy(), out["mask"].cpu().numpy().astype(bool)
    del model; torch.cuda.empty_cache()
    p = pts[mask & valid] @ cam_to_robot_R().T
    if len(p) < 50:
        raise SystemExit("mask 内有效深度点太少, 用 --height 手动给高度")
    lo, hi = np.percentile(p[:, 2], [3, 97])
    return float(hi - lo)


def normalize(mesh, height, up="y"):
    """生成的形状 -> z 朝上, 包围盒中心在原点, 高度 = height (m)"""
    mesh = mesh.copy()
    if up == "y":                                        # y 朝上 -> z 朝上: 绕 x 轴转 +90°
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    lo, hi = mesh.bounds
    mesh.apply_translation(-(lo + hi) / 2)
    mesh.apply_scale(height / (hi[2] - lo[2]))
    return mesh


def convex_parts(mesh, threshold=0.05):
    """CoACD 近似凸分解: 每块都是凸包 (MuJoCo 碰撞只认凸体)"""
    import coacd
    res = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=threshold)
    return [trimesh.Trimesh(v, f).convex_hull for v, f in res]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--hoi-json", required=True)
    ap.add_argument("--hoi-repo", required=True, help="frames.py 所在目录")
    ap.add_argument("--name", required=True, help="物体名, 输出到 assets/<name>")
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--height", type=float, default=None, help="真实高度 (m); 不给就用 MoGe 量")
    ap.add_argument("--mass", type=float, default=0.1, help="物体质量 (kg)")
    ap.add_argument("--up", default="y", choices=["y", "z"], help="生成形状的朝上轴 (Hunyuan3D 是 y)")
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
    rgba = rgba_crop(rgb, mask)
    cv2.imwrite(os.path.join(out, "crop.png"), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

    height = args.height if args.height else measure_height(rgb, mask)
    print(f"高度 {height * 100:.1f} cm ({'给定' if args.height else 'MoGe'})")
    mesh = normalize(generate_shape(rgba), height, args.up)
    mesh.export(os.path.join(out, "mesh.obj"))

    parts = convex_parts(mesh)
    for k, part in enumerate(parts):
        part.export(os.path.join(out, f"collision_{k}.obj"))
    color = np.median(rgb[mask], 0) / 255
    meta = {"size": (mesh.bounds[1] - mesh.bounds[0]).round(4).tolist(),
            "rgba": [*color.round(3).tolist(), 1.0], "frame": int(i),
            "n_collision": len(parts), "density": round(args.mass / sum(p.volume for p in parts), 1),
            "height_source": "given" if args.height else "moge"}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print("saved", out, meta)


if __name__ == "__main__":
    main()
