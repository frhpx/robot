"""
HOI-DETR 视频 demo 的命令行入口 (demo/demo_video.py 只能改文件顶部变量, 没有参数)

在 HOI-DETR 的环境 (codetr: py3.7, torch 1.11, mmcv-full 1.5) 里, 以 HOI-DETR 仓库为 cwd 运行:
  python /path/to/robot/hoi_detr_video.py --video in.mp4 --out-json out.json
复用 demo_video.process_video, 所以 json 格式和 demo 输出完全一致 (附录 C).
"""
import argparse, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--out-json", required=True)
ap.add_argument("--out-video", default=None, help="带检测框的视频, 默认和 json 同目录")
ap.add_argument("--hoi-detr", default=os.getcwd(), help="HOI-DETR 仓库根目录")
ap.add_argument("--config", default=None)
ap.add_argument("--checkpoint", default=None)
ap.add_argument("--score-thr", type=float, default=0.45)   # 样例 1.json 用的是 0.45
ap.add_argument("--nms-iou", type=float, default=0.5)
ap.add_argument("--stride", type=int, default=1)
args = ap.parse_args()

video = os.path.abspath(args.video)
out_json = os.path.abspath(args.out_json)
out_video = os.path.abspath(args.out_video or os.path.splitext(out_json)[0] + "_det.mp4")
os.chdir(args.hoi_detr)                                   # config / checkpoint 是相对仓库根的路径
sys.path[:0] = [args.hoi_detr, os.path.join(args.hoi_detr, "demo")]

import demo_video as DV
from mmdet.apis import init_detector
from mmdet.datasets.pipelines import Compose

DV.SCORE_THR, DV.NMS_IOU, DV.FRAME_STRIDE = args.score_thr, args.nms_iou, args.stride
DV.VERBOSE_LABELS = False
model = init_detector(args.config or DV.MODEL_CONFIG, args.checkpoint or DV.CHECKPOINT, device=DV.DEVICE)
model.eval()
pipe = Compose(model.cfg.data.test.pipeline)
branch = DV.find_interaction_branch(model.query_head)
tmp = os.path.splitext(out_json)[0] + "._frame.jpg"
ok = DV.process_video(video, out_video, out_json, model, pipe, branch, tmp)
if os.path.exists(tmp):
    os.remove(tmp)
if not ok:
    sys.exit("HOI-DETR failed on " + video)
print("saved", out_json)
