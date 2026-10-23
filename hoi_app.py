"""
网页: 上传视频 -> HOI-DETR -> hoi_to_mujoco.py -> 输出并排视频 (左: 原视频+检测框, 右: G1 复现)

运行 (在装了 HOI-DETR 的 Ubuntu 机器上):
  pip install gradio opencv-python imageio imageio-ffmpeg mujoco
  python hoi_app.py            # 浏览器打开 http://127.0.0.1:7860

先改下面的路径和 HOI_PY (装了 mmcv-full 的那个环境的 python).
已经有 json 的话, 网页上直接一起上传, 会跳过 HOI-DETR.
勾选 "MoGe 深度" 需要本环境装好 MoGe (见 moge_depth.py).
填了物体名: assets/<物体名> 已存在就直接用, 不存在就先跑 object_asset.py 生成 (生成一次, 之后反复用).
"""
import os, re, shutil, subprocess, sys, tempfile
import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))
HOI_REPO = os.path.expanduser("~/code/handbench/hoi")   # frames.py 所在目录 (HOI-pipeline-for-hand-detection)
MENAGERIE = os.path.expanduser("~/mujoco_menagerie")
HOI_DETR = os.path.expanduser("~/code/handbench/hoidetr")
HOI_PY = os.path.expanduser("~/miniconda3/envs/codetr/bin/python")   # HOI-DETR 环境的 python
SCORE_THR = 0.45            # 和样例 data/1.json 一致
# demo/demo_video.py 没有命令行参数, 靠模块常量指定输入/输出目录; 这里改常量后调用 main()
HOI_SNIPPET = ("import sys; sys.path.insert(0, 'demo'); import demo_video as m; "
               "m.INPUT_DIR, m.OUTPUT_DIR, m.SCORE_THR = sys.argv[1], sys.argv[2], float(sys.argv[3]); m.main()")


def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    log = f"$ {' '.join(cmd)}\n{p.stdout}{p.stderr}"
    if p.returncode != 0:
        raise gr.Error(f"命令失败, 看日志:\n{log[-1500:]}")
    return log


def pipeline(video, hoi_json, seg, use_moge, obj_name, cam_pitch):
    if video is None:
        raise gr.Error("先上传视频")
    work = tempfile.mkdtemp(prefix="hoi_app_")
    log = ""

    # 1. HOI-DETR (上传了 json 就跳过)
    if hoi_json is None:
        in_dir, out_dir = os.path.join(work, "in"), os.path.join(work, "hoi")
        os.makedirs(in_dir)
        shutil.copy(video, os.path.join(in_dir, "clip.mp4"))
        log += run([HOI_PY, "-c", HOI_SNIPPET, in_dir, out_dir, str(SCORE_THR)], cwd=HOI_DETR)
        hoi_json = os.path.join(out_dir, "clip.json")

    # 2. 机器人复现
    base = [sys.executable, os.path.join(HERE, "hoi_to_mujoco.py"),
            "--hoi-json", hoi_json, "--hoi-repo", HOI_REPO, "--menagerie", MENAGERIE,
            "--cam-pitch", str(50 if cam_pitch is None else cam_pitch)]
    seg_log = run(base + ["--list"])
    log += seg_log
    if not re.search(r"seg \d+:", seg_log):
        raise gr.Error("没有检测到右手接触段, 无法复现")
    if use_moge:
        npz = os.path.join(work, "moge.npz")
        log += run([sys.executable, os.path.join(HERE, "moge_depth.py"), "--video", video,
                    "--hoi-json", hoi_json, "--hoi-repo", HOI_REPO, "--out", npz])
        base += ["--depth-npz", npz]
    obj_name = (obj_name or "").strip()
    if obj_name:
        obj_dir = os.path.join(HERE, "assets", obj_name)
        if not os.path.exists(os.path.join(obj_dir, "meta.json")):
            log += run([sys.executable, os.path.join(HERE, "object_asset.py"), "--video", video,
                        "--hoi-json", hoi_json, "--hoi-repo", HOI_REPO, "--name", obj_name])
        base += ["--object", obj_dir]
    out = os.path.join(work, "result.mp4")
    cmd = base + ["--video", out, "--src-video", video]
    if seg is not None and seg >= 0:
        cmd += ["--seg", str(int(seg))]
    log += run(cmd, cwd=HERE)
    return out, log


with gr.Blocks(title="HOI -> G1") as demo:
    gr.Markdown("## 上传视频 → HOI 检测 → G1 机器人复现")
    with gr.Row():
        with gr.Column():
            video = gr.Video(label="输入视频", sources=["upload"])
            hoi_json = gr.File(label="(可选) 已有的 HOI json, 上传则跳过检测", file_types=[".json"], type="filepath")
            seg = gr.Number(label="接触段编号 (-1 = 自动选最长)", value=-1, precision=0)
            use_moge = gr.Checkbox(label="MoGe 深度 (不勾选 = 用手框宽度估深度)", value=True)
            obj_name = gr.Textbox(label="物体名 (留空 = 立方体; 已有资产直接用, 没有就从视频生成)", value="")
            cam_pitch = gr.Number(label="相机下倾角 (度): 头戴约 50, 桌边侧拍约 15", value=50)
            btn = gr.Button("运行", variant="primary")
        with gr.Column():
            result = gr.Video(label="结果 (左: 原视频+检测框, 右: G1)")
            log = gr.Textbox(label="日志", lines=12)
    btn.click(pipeline, [video, hoi_json, seg, use_moge, obj_name, cam_pitch], [result, log])

if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")      # 无窗口渲染
    demo.launch()
