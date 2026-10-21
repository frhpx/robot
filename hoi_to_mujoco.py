"""
HOI pipeline 输出 -> MuJoCo G1 右手复现 (物体用立方体代替)

输入: HOI-DETR 的 json (经 frames.FrameBoxes 分好左右手) + json 里的 hf 接触链接
流程:
  1. 每帧取右手框: 中心 (u, v) + 宽度 w
  2. 针孔反投影: 深度 d = f * HAND_WIDTH / w (手框越大离相机越近), 得到相机坐标系 3D 点
  3. 相机 -> 机器人坐标: 头部相机向下倾斜 CAM_PITCH; 再平移, 使 "接触开始那一帧" 的手正好在方块上
  4. 接触 (hf) 去抖 -> 手指开合; 握紧时方块跟着抓取点走, 松开后落回桌面
  5. IK (阻尼最小二乘, 同 ik_pick_place.py) 让抓取点跟踪这条轨迹

用法:
  python hoi_to_mujoco.py --hoi-json ../HOI-pipeline-for-hand-detection/data/1.json --menagerie C:/Users/30691/mujoco_menagerie
  python hoi_to_mujoco.py ... --list                 # 列出所有接触段
  python hoi_to_mujoco.py ... --seg 3                # 复现第 3 段 (默认最长的一段)
  python hoi_to_mujoco.py ... --video out.mp4        # 不开窗口, 离屏渲染成视频
  python hoi_to_mujoco.py ... --video out.mp4 --src-video in.mp4   # 左: 原视频+检测框, 右: 机器人
"""
import argparse, os, sys, json, time
import numpy as np
import mujoco

# ---------- 可调参数 ----------
HAND_WIDTH = 0.09          # 真实手掌宽度 (m), 用来从框宽估深度
D_CONTACT = 0.45           # 接触时手到头部相机的距离 (m), 定下焦距 f
CAM_PITCH = np.deg2rad(50) # 头部相机向下倾斜角
CUBE_POS = np.array([0.33, -0.20, 0.765])   # 方块中心 (桌上, 右手够得到)
CUBE_HALF = 0.025
TABLE_TOP = CUBE_POS[2] - CUBE_HALF
WS_LO = np.array([0.15, -0.45, CUBE_POS[2]])          # 工作空间裁剪, 防止 IK 发散
WS_HI = np.array([0.50,  0.05, CUBE_POS[2] + 0.30])
SMOOTH = 15                # 轨迹平滑窗口 (帧)
PAD = 30                   # 接触段前后各多放多少帧 (看到接近和离开)

BASE = [0, 0, 0.793, 1, 0, 0, 0]
ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
       "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
FINGERS = {"right_hand_thumb_1_joint": 0.3, "right_hand_thumb_2_joint": -0.4,
           "right_hand_index_0_joint": 0.5, "right_hand_index_1_joint": 0.5,
           "right_hand_middle_0_joint": 0.5, "right_hand_middle_1_joint": 0.5}
GRASP_OFFSET = np.array([0.11, 0.035, 0.0])  # 抓取点, 手腕坐标系 (同 ik_pick_place.py)
q0 = np.array([-0.4, 0, 0, 0.9, 0, 0, 0])


# ================= 1. 读 HOI: 右手框 + 接触 =================
def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def load_right_hand(hoi_json, hoi_repo):
    sys.path.insert(0, hoi_repo)
    from frames import FrameBoxes
    fb = FrameBoxes(hoi_json)
    raw = json.load(open(hoi_json, encoding="utf-8"))
    n = len(fb)
    box = np.full((n, 3), np.nan)       # u, v, w
    contact = np.zeros(n)
    for fr in raw["frames"]:
        i = fr["frame_idx"]
        r = fb[i].right
        if r is None:
            continue
        box[i] = [(r[0] + r[2]) / 2, (r[1] + r[3]) / 2, r[2] - r[0]]
        dets = fr["detections"]
        for l in fr.get("hf", []):                     # hf 下标是 0 起的 detections 下标
            if l["prob"] < 0.5:
                continue
            for k in (l["a"], l["b"]):
                if 0 <= k < len(dets) and dets[k]["class_name"] == "hand" and iou(dets[k]["box"], r) > 0.7:
                    contact[i] = 1
    # 缺帧线性插值, 再滑动平均
    t = np.arange(n)
    for c in range(3):
        ok = ~np.isnan(box[:, c])
        box[:, c] = np.interp(t, t[ok], box[ok, c])
    k = np.ones(SMOOTH) / SMOOTH
    box = np.stack([np.convolve(np.pad(box[:, c], SMOOTH // 2, mode="edge"), k, "valid") for c in range(3)], 1)
    return box, contact, fb.width, fb.height, fb.fps, fb


def segments(c, gap=5, min_len=10):
    """接触 0/1 序列 -> [(start, end)]: 合并间隔 <= gap 的段, 丢掉短于 min_len 的段"""
    segs, i, n = [], 0, len(c)
    while i < n:
        if c[i]:
            j = i
            while j < n and c[j]:
                j += 1
            if segs and i - segs[-1][1] <= gap:
                segs[-1] = (segs[-1][0], j)
            else:
                segs.append((i, j))
            i = j
        else:
            i += 1
    return [s for s in segs if s[1] - s[0] >= min_len]


# ================= 2. 像素 -> 机器人坐标 =================
def to_robot(box, W, H, onset):
    u, v, w = box[:, 0], box[:, 1], box[:, 2]
    f = D_CONTACT * box[onset, 2] / HAND_WIDTH        # 选 f 使接触那帧深度 = D_CONTACT
    d = f * HAND_WIDTH / w                            # 每帧深度
    pc = np.stack([(u - W / 2) * d / f, (v - H / 2) * d / f, d], 1)   # 相机系: x右 y下 z前
    a = CAM_PITCH
    R = np.array([[0, -np.sin(a), np.cos(a)],         # 列 = 相机 x/y/z 轴在机器人系 (x前 y左 z上) 里的方向
                  [-1, 0, 0],
                  [0, -np.cos(a), -np.sin(a)]])
    p = pc @ R.T
    p = p - p[onset] + CUBE_POS                       # 接触开始时抓取点 = 方块中心
    return np.clip(p, WS_LO, WS_HI)


# ================= 3. 场景: G1 + 桌子 + 方块 =================
def build_model(menagerie):
    spec = mujoco.MjSpec.from_file(os.path.join(menagerie, "unitree_g1", "scene_with_hands.xml"))
    wb = spec.worldbody
    table = wb.add_body(name="table", pos=[CUBE_POS[0] + 0.05, -0.15, TABLE_TOP / 2])
    table.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.15, 0.25, TABLE_TOP / 2], rgba=[0.6, 0.45, 0.3, 1])
    cube = wb.add_body(name="cube", mocap=True, pos=CUBE_POS.tolist())
    cube.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3, rgba=[0.9, 0.2, 0.2, 1],
                  contype=0, conaffinity=0)
    return spec.compile()


# ================= 4. 主循环 =================
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--hoi-json", default=os.path.join(here, "..", "HOI-pipeline-for-hand-detection", "data", "1.json"))
    ap.add_argument("--hoi-repo", default=None, help="frames.py 所在目录, 默认 = json 的上一级")
    ap.add_argument("--menagerie", default=r"C:\Users\30691\mujoco_menagerie")
    ap.add_argument("--seg", type=int, default=None, help="复现第几段接触 (默认最长)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--video", default=None, help="离屏渲染到 mp4, 不开窗口")
    ap.add_argument("--src-video", default=None, help="和 json 对应的原视频; 给了就输出左右并排")
    args = ap.parse_args()
    hoi_repo = args.hoi_repo or os.path.dirname(os.path.dirname(os.path.abspath(args.hoi_json)))

    box, contact, W, H, fps, fb = load_right_hand(args.hoi_json, hoi_repo)
    segs = segments(contact)
    if args.list or not segs:
        for i, (s, e) in enumerate(segs):
            print(f"seg {i}: frame {s}-{e}  ({(e - s) / fps:.1f}s)")
        return
    si = args.seg if args.seg is not None else max(range(len(segs)), key=lambda i: segs[i][1] - segs[i][0])
    s, e = segs[si]
    a, b = max(0, s - PAD), min(len(box), e + PAD)
    traj = to_robot(box, W, H, s)[a:b]
    grip = np.zeros(len(box)); grip[s:e] = 1
    k = np.ones(9) / 9
    grip = np.convolve(np.pad(grip, 4, mode="edge"), k, "valid")[a:b]      # 手指开合平滑
    print(f"seg {si}: frame {s}-{e}, 回放 {a}-{b}")

    m = build_model(args.menagerie)
    d = mujoco.MjData(m)
    qa = [m.joint(n).qposadr[0] for n in ARM]
    va = [m.joint(n).dofadr[0] for n in ARM]
    fa = [m.joint(n).qposadr[0] for n in FINGERS]
    f_closed = np.array(list(FINGERS.values()))
    hand = m.body("right_wrist_yaw_link").id
    cube = m.body("cube").mocapid[0]

    def grasp_point():
        return d.xpos[hand] + d.xmat[hand].reshape(3, 3) @ GRASP_OFFSET

    def ik(target):
        d.qpos[qa] += 0.05 * (q0 - d.qpos[qa])
        for _ in range(10):
            mujoco.mj_forward(m, d)
            p = grasp_point()
            Jp = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, Jp, None, p, hand)
            J = Jp[:, va]
            d.qpos[qa] += J.T @ np.linalg.solve(J @ J.T + 0.05 ** 2 * np.eye(3), target - p)

    # 起始: 先让手臂收敛到第一帧目标, 避免第一帧跳变
    d.qpos[:7] = BASE; d.qpos[qa] = q0
    for _ in range(20):
        ik(traj[0])

    def run(show):
        d.mocap_pos[cube] = CUBE_POS
        holding, vz, err = False, 0.0, []
        for t in range(len(traj)):
            ik(traj[t])
            d.qpos[fa] = grip[t] * f_closed
            mujoco.mj_forward(m, d)
            gp = grasp_point()
            err.append(np.linalg.norm(gp - traj[t]))
            if grip[t] > 0.9 and np.linalg.norm(gp - d.mocap_pos[cube]) < 0.06:
                holding = True                       # 握紧且手在方块旁 -> 抓住
            elif grip[t] < 0.5:
                holding = False
            if holding:
                d.mocap_pos[cube] = gp; vz = 0.0
            elif d.mocap_pos[cube][2] > CUBE_POS[2]:  # 松手 -> 自由落体到桌面
                vz -= 9.81 / fps
                d.mocap_pos[cube][2] = max(CUBE_POS[2], d.mocap_pos[cube][2] + vz / fps)
            mujoco.mj_forward(m, d)
            if not show(t, holding):
                break
        print(f"IK 跟踪误差: mean {np.mean(err) * 100:.1f} cm, max {np.max(err) * 100:.1f} cm")

    if args.video:
        import imageio
        r = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera(); cam.lookat[:] = [0.25, -0.15, 0.85]
        cam.distance, cam.azimuth, cam.elevation = 1.3, 150, -25
        frames = []

        src = None
        if args.src_video:
            import cv2
            src = cv2.VideoCapture(args.src_video)
            src.set(cv2.CAP_PROP_POS_FRAMES, a)

        def show(t, holding):
            r.update_scene(d, cam); img = r.render()
            if src is not None:
                ok, raw = src.read()
                if ok:
                    raw = fb[a + t].draw(raw)
                    txt = "contact" if grip[t] > 0.5 else ""
                    cv2.putText(raw, f"frame {a + t} {txt}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                    raw = cv2.resize(raw, (raw.shape[1] * 480 // raw.shape[0], 480))
                    img = np.hstack([raw, img])
            frames.append(img); return True
        run(show)
        imageio.mimsave(args.video, frames, fps=fps)
        print("saved", args.video)
    else:
        from mujoco import viewer as mjv
        with mjv.launch_passive(m, d) as viewer:
            def show(t, holding):
                viewer.sync(); time.sleep(1 / fps); return viewer.is_running()
            while viewer.is_running():
                run(show)
                time.sleep(0.5)


if __name__ == "__main__":
    main()
