"""
网页: 上传视频 -> HOI-DETR -> hoi_to_mujoco.py -> 输出并排视频 (左: 原视频+检测框, 右: G1 复现)

运行 (在装了 HOI-DETR 的 Ubuntu 机器上):
  pip install gradio opencv-python imageio imageio-ffmpeg mujoco
  python hoi_app.py            # 浏览器打开 http://127.0.0.1:7860

先改下面三个路径, 和 HOI_CMD (你平时跑 HOI-DETR 视频 demo 的那条命令).
已经有 json 的话, 网页上直接一起上传, 会跳过 HOI-DETR.
"""
import os, re, shlex, subprocess, sys, tempfile
import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))
HOI_REPO = os.path.expanduser("~/code/handbench/hoi")   # frames.py 所在目录 (HOI-pipeline-for-hand-detection)
MENAGERIE = os.path.expanduser("~/mujoco_menagerie")
HOI_DETR = os.path.expanduser("~/code/handbench/hoidetr")
# {video} = 输入视频, {json} = 要写出的 json; 按你自己的 demo 脚本改
# demo/demo_video.py 没有命令行参数, 用 hoi_detr_video.py 包一层; HOI-DETR 在 codetr 环境 (mmcv-full) 里跑
CONDA = os.path.expanduser("~/miniconda3/bin/conda")
HOI_CMD = (f"{shlex.quote(CONDA)} run -n codetr python {shlex.quote(os.path.join(HERE, 'hoi_detr_video.py'))}"
           " --video {video} --out-json {json}")


def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    log = f"$ {' '.join(cmd)}\n{p.stdout}{p.stderr}"
    if p.returncode != 0:
        raise gr.Error(f"命令失败, 看日志:\n{log[-1500:]}")
    return log


def pipeline(video, hoi_json, seg):
    if video is None:
        raise gr.Error("先上传视频")
    work = tempfile.mkdtemp(prefix="hoi_app_")
    log = ""

    # 1. HOI-DETR (上传了 json 就跳过)
    if hoi_json is None:
        hoi_json = os.path.join(work, "hoi.json")
        cmd = shlex.split(HOI_CMD.format(video=shlex.quote(video), json=shlex.quote(hoi_json)))
        log += run(cmd, cwd=HOI_DETR)

    # 2. 机器人复现
    base = [sys.executable, os.path.join(HERE, "hoi_to_mujoco.py"),
            "--hoi-json", hoi_json, "--hoi-repo", HOI_REPO, "--menagerie", MENAGERIE]
    seg_log = run(base + ["--list"])
    log += seg_log
    if not re.search(r"seg \d+:", seg_log):
        raise gr.Error("没有检测到右手接触段, 无法复现")
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
            btn = gr.Button("运行", variant="primary")
        with gr.Column():
            result = gr.Video(label="结果 (左: 原视频+检测框, 右: G1)")
            log = gr.Textbox(label="日志", lines=12)
    btn.click(pipeline, [video, hoi_json, seg], [result, log])

if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")      # 无窗口渲染
    demo.launch()
