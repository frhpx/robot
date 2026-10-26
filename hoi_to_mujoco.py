"""
HOI pipeline 输出 -> MuJoCo G1 右手复现 (物体: 立方体, 或 --object: object_points.py 的点云 / object_asset.py 的 mesh)

输入: HOI-DETR 的 json (经 frames.FrameBoxes 分好左右手) + json 里的 hf 接触链接
流程:
  1. 每帧取右手框: 中心 (u, v) + 宽度 w
  2. 手的 3D 点 (相机系): 有 --depth-npz (moge_depth.py 输出) 就用 MoGe 的米制点图;
     否则用针孔反投影 d = f * HAND_WIDTH / w (手框越大离相机越近)
  3. 相机 -> 机器人坐标: 物体资产有 scene.json (object_points.py 用桌面平面估的相机姿态) 且有 --depth-npz 时,
     手和物体用同一个 R_cam2table 转到桌面系, 再整体平移一次 (物体落在 OBJ_XY, 桌面 = TABLE_TOP), 保留真实相对位置;
     否则 (后备): 头部相机向下倾斜 --cam-pitch, 再平移使 "接触开始那一帧" 的手正好在物体上
  4. 接触 (hf) 去抖 -> 抓取事件: 接触段内先让抓取点对准物体的实际位置, 到位后手指合拢,
     抓稳后跟随人手轨迹的相对运动; 接触段结束 -> 手指张开
  5. IK: 阻尼最小二乘 (位置 + 固定的抓取朝向) + 零空间姿态正则 + 关节速度限制 (保证手臂连续);
     接触前离物体越近, 目标点最低高度越高 -> 手从上方落到物体上, 不会横扫碰倒
  6. 物理仿真: 物体和桌子都是有碰撞的刚体; 手臂/手指是位置电机 + 重力补偿;
     拇指和手指同时碰到物体 -> 打开焊接约束稳住抓取 (同 ik_grasp_drop.py), 松手 -> 关掉, 物体自由下落
     --no-weld: 不用焊接, 完全靠摩擦抓

用法:
  python hoi_to_mujoco.py --hoi-json ../HOI-pipeline-for-hand-detection/data/1.json --menagerie C:/Users/30691/mujoco_menagerie
  python hoi_to_mujoco.py ... --list                 # 列出所有接触段
  python hoi_to_mujoco.py ... --seg 3                # 复现第 3 段 (默认最长的一段)
  python hoi_to_mujoco.py ... --video out.mp4        # 不开窗口, 离屏渲染成视频
  python hoi_to_mujoco.py ... --video out.mp4 --src-video in.mp4   # 左: 原视频+检测框, 右: 机器人
  python hoi_to_mujoco.py ... --depth-npz moge.npz   # 用 MoGe 深度
  python hoi_to_mujoco.py ... --object assets/mug     # 用物体资产代替立方体
  python hoi_to_mujoco.py ... --object assets/mug --obj-yaw 90   # 物体绕竖直轴转 90° (窄边对着抓握方向)
"""
import argparse, os, sys, json, time
import numpy as np
import mujoco

# ---------- 可调参数 ----------
HAND_WIDTH = 0.09          # 真实手掌宽度 (m), 用来从框宽估深度
D_CONTACT = 0.45           # 接触时手到头部相机的距离 (m), 定下焦距 f
CAM_PITCH = np.deg2rad(50) # 头部相机向下倾斜角
TABLE_TOP = 0.74
OBJ_XY = np.array([0.33, -0.20])            # 物体放在桌上的位置 (右手够得到)
CUBE_HALF = 0.025                           # 没给 --object 时的立方体
CUBE_MASS = 0.1
OBJ_POS = np.array([*OBJ_XY, TABLE_TOP + CUBE_HALF])  # 物体中心; --object 时按物体高度重设
WS_LO = np.array([0.15, -0.45, TABLE_TOP + 0.025])    # 工作空间裁剪: x 上限 = G1 右手贴桌面能够到的距离
WS_HI = np.array([0.42,  0.05, TABLE_TOP + 0.30])
SMOOTH = 15                # 轨迹平滑窗口 (帧)
K_NULL = 0.2               # 零空间里往 q0 拉的强度
MAX_VEL = 3.0              # 关节最大速度 (rad/s)
W_ROT = 0.3                # IK 里手腕朝向误差的权重 (m/rad)
HOVER = 0.10               # 接触前: 离物体水平距离 > 2.5 cm 时, 抓取点至少比物体中心高这么多
PAD = 30                   # 接触段前后各多放多少帧 (看到接近和离开)

BASE = [0, 0, 0.793, 1, 0, 0, 0]
ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
       "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
FINGERS = ["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
           "right_hand_index_0_joint", "right_hand_index_1_joint",
           "right_hand_middle_0_joint", "right_hand_middle_1_joint"]
F_CLOSED = np.array([0, 0.3, -0.8, 1.2, 1.2, 1.2, 1.2])   # 手指握紧的目标角度 (同 ik_grasp_drop.py)
GRASP_OFFSET = np.array([0.12, 0.05, 0.0])  # 抓取点: 拇指和手指之间, 手腕坐标系 (同 ik_grasp_drop.py)
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
        sided = [det["box"] for det in fr["detections"] if det.get("side") == "Right"]
        if any("side" in det for det in fr["detections"]):   # json 自带左右手 (hoi_cpu.py): 直接用
            r = sided[0] if sided else None
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
def fill(x):
    """(N, C) 里的 NaN 行按时间线性插值"""
    t = np.arange(len(x))
    for c in range(x.shape[1]):
        ok = ~np.isnan(x[:, c])
        x[:, c] = np.interp(t, t[ok], x[ok, c])
    return x


def smooth(x, win=SMOOTH):
    """先 5 帧中值滤波去离群点, 再滑动平均"""
    p = np.pad(x, ((2, 2), (0, 0)), mode="edge")
    x = np.median(np.stack([p[i:i + len(x)] for i in range(5)]), 0)
    k = np.ones(win) / win
    return np.stack([np.convolve(np.pad(x[:, c], win // 2, mode="edge"), k, "valid") for c in range(x.shape[1])], 1)


def cam_points_from_box(box, W, H, onset):
    """没有深度图时: 用框宽估深度, 返回相机系 3D 点 (x右 y下 z前)"""
    box = smooth(fill(box.copy()))                    # 1/w 会放大噪声, 先在像素域平滑
    u, v, w = box[:, 0], box[:, 1], box[:, 2]
    f = D_CONTACT * w[onset] / HAND_WIDTH             # 选 f 使接触那帧深度 = D_CONTACT
    d = f * HAND_WIDTH / w
    return np.stack([(u - W / 2) * d / f, (v - H / 2) * d / f, d], 1)


def cam_points_from_moge(npz, n):
    """moge_depth.py 的输出: 每帧右手在相机系的米制 3D 点"""
    z = np.load(npz)
    pc = np.full((n, 3), np.nan)
    ok = z["valid"].astype(bool)
    pc[z["frame"][ok]] = z["hand_xyz"][ok]
    return fill(pc)


def cam_to_robot_R(a=None):
    """相机系 (x右 y下 z前) -> 机器人系 (x前 y左 z上); 列 = 相机各轴在机器人系里的方向"""
    a = CAM_PITCH if a is None else a
    return np.array([[0, -np.sin(a), np.cos(a)],
                     [-1, 0, 0],
                     [0, -np.cos(a), -np.sin(a)]])


def scene_shift(meta):
    """桌面系 -> 世界系的平移: 物体底面中心落在 OBJ_XY, 桌面 (z = 0) 落在 TABLE_TOP"""
    pose = np.array(meta["pose_table"])
    return np.array([OBJ_XY[0] - pose[0], OBJ_XY[1] - pose[1], TABLE_TOP])


def cam_to_world(pc, scene, meta):
    """相机系点 -> 世界系, 用 scene.json 的桌面系 (和物体同一个变换)"""
    R, p0 = np.array(scene["R_cam2table"]), np.array(scene["p_table"])
    return (pc - p0) @ R.T + scene_shift(meta)


def world_to_cam(p, scene, meta):
    R, p0 = np.array(scene["R_cam2table"]), np.array(scene["p_table"])
    return (p - scene_shift(meta)) @ R + p0


def to_robot(pc, onset):
    """后备 (没有 scene.json): --cam-pitch 旋转, 接触开始时的手平移到物体中心"""
    p = pc @ cam_to_robot_R().T
    p = p - p[onset] + OBJ_POS                        # 接触开始时抓取点 = 物体中心
    return smooth(np.clip(p, WS_LO, WS_HI))           # 先裁剪再平滑, 避免折线


# ================= 3. 场景: G1 + 桌子 + 方块 =================
def build_model(menagerie, obj_dir=None, obj_yaw=0.0):
    """G1 + 桌子 + 物体 (自由刚体, 有碰撞和重力) + 抓取焊接约束 (默认关闭)
       obj_dir: object_points.py / object_asset.py 的输出目录; None = 立方体
       obj_yaw: 物体绕竖直轴的初始转角 (rad)"""
    spec = mujoco.MjSpec.from_file(os.path.join(menagerie, "unitree_g1", "scene_with_hands.xml"))
    wb = spec.worldbody
    table = wb.add_body(name="table", pos=[OBJ_XY[0] + 0.05, -0.15, TABLE_TOP / 2])
    table.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.15, 0.25, TABLE_TOP / 2], rgba=[0.6, 0.45, 0.3, 1])
    obj = wb.add_body(name="obj", pos=OBJ_POS.tolist(), quat=[np.cos(obj_yaw / 2), 0, 0, np.sin(obj_yaw / 2)])
    obj.add_freejoint(name="obj_joint")
    meta = json.load(open(os.path.join(obj_dir, "meta.json"))) if obj_dir else None
    if obj_dir is None:
        obj.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3, rgba=[0.9, 0.2, 0.2, 1], mass=CUBE_MASS)
    elif meta.get("type") == "points":                                        # 点云: 每个点一个小球, 同一个刚体
        pts = np.load(os.path.join(obj_dir, "points.npy"))
        cols = np.load(os.path.join(obj_dir, "colors.npy"))
        for p, c in zip(pts, cols):
            obj.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[meta["radius"], 0, 0], pos=p.tolist(),
                         rgba=[*c.tolist(), 1], mass=meta["mass"] / len(pts),
                         friction=[1.0, 0.01, 0.001])
    else:
        spec.add_mesh(name="obj_vis", file=os.path.abspath(os.path.join(obj_dir, "mesh.obj")))
        obj.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="obj_vis", rgba=meta["rgba"],
                     contype=0, conaffinity=0, density=0)                     # 显示用, 不参与碰撞
        for i in range(meta["n_collision"]):                                  # 碰撞用: 凸分解的各块
            spec.add_mesh(name=f"obj_col{i}", file=os.path.abspath(os.path.join(obj_dir, f"collision_{i}.obj")))
            obj.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"obj_col{i}", group=3,
                         density=meta["density"])
    spec.add_equality(name="grasp", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                      name1="right_wrist_yaw_link", name2="obj", active=False)
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
    ap.add_argument("--depth-npz", default=None, help="moge_depth.py 的输出; 不给就用框宽估深度")
    ap.add_argument("--object", default=None, help="object_points.py / object_asset.py 生成的物体目录; 不给就用立方体")
    ap.add_argument("--obj-yaw", type=float, default=0, help="物体绕竖直轴旋转 (度), 把窄边对着抓握方向")
    ap.add_argument("--no-weld", action="store_true", help="纯摩擦抓取: 拇指和手指碰到物体后不打开焊接约束")
    ap.add_argument("--cam-pitch", type=float, default=50, help="相机向下倾斜角 (度): 头戴约 50, 桌边侧拍约 15")
    args = ap.parse_args()
    hoi_repo = args.hoi_repo or os.path.dirname(os.path.dirname(os.path.abspath(args.hoi_json)))

    global OBJ_POS, CAM_PITCH
    CAM_PITCH = np.deg2rad(args.cam_pitch)
    meta = scene = None
    if args.object:                                    # 物体中心 = 桌面 + 半个物体高度
        meta = json.load(open(os.path.join(args.object, "meta.json")))
        OBJ_POS = np.array([*OBJ_XY, TABLE_TOP + meta["size"][2] / 2])
        sp = os.path.join(args.object, "scene.json")
        scene = json.load(open(sp)) if os.path.exists(sp) and "pose_table" in meta else None
        if "cam_pitch" in meta and abs(meta["cam_pitch"] - args.cam_pitch) > 1e-6:
            print(f"警告: 物体资产是按 --cam-pitch {meta['cam_pitch']} 生成的, 现在用的是 {args.cam_pitch}, "
                  "物体的朝向会不一致")
    box, contact, W, H, fps, fb = load_right_hand(args.hoi_json, hoi_repo)
    segs = segments(contact)
    if args.list or not segs:
        for i, (s, e) in enumerate(segs):
            print(f"seg {i}: frame {s}-{e}  ({(e - s) / fps:.1f}s)")
        return
    si = args.seg if args.seg is not None else max(range(len(segs)), key=lambda i: segs[i][1] - segs[i][0])
    s, e = segs[si]
    a, b = max(0, s - PAD), min(len(box), e + PAD)
    pc = cam_points_from_moge(args.depth_npz, len(box)) if args.depth_npz else cam_points_from_box(box, W, H, s)
    if scene is not None and args.depth_npz:           # 真实相对位置: 手和物体同一个变换
        pw = cam_to_world(pc, scene, meta)
        print(f"桌面系 (scene.json, 隐含下倾角 {scene['pitch_deg']}°): 接触开始 (第 {s} 帧) 手到物体中心 "
              f"{np.linalg.norm(pw[s] - OBJ_POS) * 100:.1f} cm")
        traj = smooth(np.clip(pw, WS_LO, WS_HI))[a:b]
    else:
        if scene is not None:
            print("没有 --depth-npz, 手轨迹没有米制深度: 退回 --cam-pitch + 接触时平移")
        traj = to_robot(pc, s)[a:b]
    for t in range(s - a):                             # 接触前: 从上方接近物体
        dxy = np.linalg.norm(traj[t, :2] - OBJ_POS[:2])
        traj[t, 2] = max(traj[t, 2], OBJ_POS[2] + HOVER * np.clip((dxy - 0.005) / 0.02, 0, 1))
    traj = smooth(traj, 5)
    in_contact = np.zeros(b - a, bool); in_contact[s - a:e - a] = True
    print(f"seg {si}: frame {s}-{e}, 回放 {a}-{b}")

    m = build_model(args.menagerie, args.object, np.deg2rad(args.obj_yaw))
    d = mujoco.MjData(m)            # 物理仿真
    dk = mujoco.MjData(m)           # IK 试算 (只做运动学)
    qa = [m.joint(n).qposadr[0] for n in ARM]
    va = [m.joint(n).dofadr[0] for n in ARM]
    fq = [m.joint(n).qposadr[0] for n in FINGERS]
    arm_act = [m.actuator(n).id for n in ARM]
    finger_act = [m.actuator(n).id for n in FINGERS]
    adof = m.jnt_dofadr[m.actuator_trnid[:, 0]]       # 每个电机对应的自由度 (重力补偿用)
    kp = m.actuator_gainprm[:, 0]
    hand = m.body("right_wrist_yaw_link").id
    obj = m.body("obj").id
    obj_q = m.joint("obj_joint").qposadr[0]
    weld = m.equality("grasp").id
    thumb = {m.body(f"right_hand_thumb_{i}_link").id for i in range(3)}
    finger = {m.body(f"right_hand_{f}_{i}_link").id for f in ("index", "middle") for i in range(2)}
    n_sub = max(1, round(1 / (fps * m.opt.timestep)))  # 每个视频帧走几步物理

    def grasp_point(data):
        return data.xpos[hand] + data.xmat[hand].reshape(3, 3) @ GRASP_OFFSET

    lo, hi = m.jnt_range[[m.joint(n).id for n in ARM]].T
    max_dq = MAX_VEL / fps

    q_goal = np.zeros(4)                               # 手腕目标朝向 (下面算)

    def ik(target, rot=True):
        """在 dk 上算: 位置 + 手腕朝向; 从上一帧解出发 (warm start); 冗余自由度在零空间里拉向 q0"""
        q_prev = dk.qpos[qa].copy()
        for _ in range(10):
            mujoco.mj_kinematics(m, dk); mujoco.mj_comPos(m, dk)
            p = grasp_point(dk)
            Jp, Jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jac(m, dk, Jp, Jr, p, hand)
            e = target - p
            J = Jp[:, va]
            if rot:                                    # 朝向误差 (轴角), 权重 W_ROT
                qc, dq_rot = np.zeros(4), np.zeros(3)
                mujoco.mju_negQuat(qc, dk.xquat[hand]); mujoco.mju_mulQuat(qc, q_goal, qc)
                mujoco.mju_quat2Vel(dq_rot, qc, 1.0)
                e, J = np.r_[e, W_ROT * dq_rot], np.r_[J, W_ROT * Jr[:, va]]
            Jinv = J.T @ np.linalg.inv(J @ J.T + 0.05 ** 2 * np.eye(len(e)))
            N = np.eye(7) - Jinv @ J                                  # 零空间投影
            dk.qpos[qa] += Jinv @ e + N @ (K_NULL * (q0 - dk.qpos[qa]))
        q = q_prev + np.clip(dk.qpos[qa] - q_prev, -max_dq, max_dq)   # 关节速度限制
        dk.qpos[qa] = np.clip(q, lo, hi)

    def touching():
        """物体是否同时碰到拇指和 (食指或中指)"""
        t = f = False
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            other = b2 if b1 == obj else b1 if b2 == obj else None
            t |= other in thumb
            f |= other in finger
        return t and f

    def set_weld():
        """打开焊接: 物体按当前相对位姿固定在手腕上"""
        R = d.xmat[hand].reshape(3, 3)
        m.eq_data[weld, 0:3] = 0                      # anchor: MjSpec 编译后默认是 [0, 1, 0], 不清零会把物体拽飞 1 m
        m.eq_data[weld, 3:6] = R.T @ (d.xpos[obj] - d.xpos[hand])
        qinv, rel = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(qinv, d.xquat[hand]); mujoco.mju_mulQuat(rel, qinv, d.xquat[obj])
        m.eq_data[weld, 6:10] = rel
        d.eq_active[weld] = 1

    def step(arm_q, finger_q):
        """一个视频帧: 位置电机 + 重力补偿, 骨盆固定"""
        target = np.zeros(m.nu)
        target[arm_act], target[finger_act] = arm_q, finger_q
        for _ in range(n_sub):
            d.ctrl[:] = target + d.qfrc_bias[adof] / kp
            d.qpos[:7] = BASE; d.qvel[:6] = 0
            mujoco.mj_step(m, d)

    # 抓取朝向: 从 q0 出发只做位置 IK 到物体中心上方 1 cm, 记下手腕朝向 (同 ik_grasp_drop.py 的抓法)
    dk.qpos[:7] = BASE; dk.qpos[qa] = q0
    max_dq_run, max_dq = max_dq, np.inf
    for _ in range(50):
        ik(OBJ_POS + [0, 0, 0.01], rot=False)
    q_goal[:] = dk.xquat[hand]
    # 起始姿势: 让手臂收敛到第一帧目标 (不限速)
    dk.qpos[qa] = q0
    for _ in range(50):
        ik(traj[0])
    max_dq = max_dq_run
    q_start = dk.qpos[qa].copy()

    def reset():
        mujoco.mj_resetData(m, d)
        d.qpos[:7] = BASE; d.qpos[qa] = q_start
        d.qpos[obj_q:obj_q + 7] = m.qpos0[obj_q:obj_q + 7]      # = OBJ_POS + --obj-yaw 的朝向
        d.eq_active[weld] = 0
        dk.qpos[:] = d.qpos
        mujoco.mj_forward(m, d)
        for _ in range(10):                            # 先让物体在桌上落稳 (约 0.3 s)
            step(q_start, np.zeros(len(FINGERS)))

    def run(show):
        reset()
        held, err, z = None, [], []
        close = 0.0                 # 手指合拢程度 0~1
        grasped_at = None           # 抓稳那一帧
        target = traj[0]
        offset = np.zeros(3)        # 抓稳后: 机器人目标 = 人手轨迹 + offset (保证不跳变)
        last_dist = np.inf
        for t in range(len(traj)):
            if in_contact[t] and grasped_at is None:   # 接触中、还没抓稳: 对准物体实际位置
                if close == 0:
                    target = d.xpos[obj] + [0, 0, 0.01]
                    dist = np.linalg.norm(grasp_point(d) - target)
                    stalled = dist < 0.04 and last_dist - dist < 0.001  # 手指碰到桌面等, 降不下去了
                    last_dist = dist
                if close > 0 or dist < 0.015 or stalled:
                    close = min(1.0, close + 1 / 8)    # 到位后约 0.27 s 合拢; 合拢期间目标锁定不动
                if held is not None or (args.no_weld and close >= 1):
                    grasped_at = t; offset = grasp_point(d) - traj[t]   # 抓稳: 之后跟人手的相对运动
            else:
                target = traj[t] + offset
                if not in_contact[t]:
                    close = max(0.0, close - 1 / 8)    # 接触段结束: 张开
            ik(target)
            f_target = close * F_CLOSED
            if held is not None:
                if not in_contact[t]:                  # 松手: 关焊接, 物体交给物理
                    d.eq_active[weld] = 0; held = None
                else:
                    f_target = held                    # 抓住后手指保持在碰到时的角度
            elif not args.no_weld and close > 0 and touching():
                set_weld(); held = d.qpos[fq].copy()
            step(dk.qpos[qa], f_target)
            err.append(np.linalg.norm(grasp_point(d) - target)); z.append(d.xpos[obj][2])
            if not show(t, held is not None):
                break
        rest = OBJ_POS[2]
        print(f"手跟踪误差: mean {np.mean(err) * 100:.1f} cm, max {np.max(err) * 100:.1f} cm")
        print(f"物体: 最高抬起 {(max(z) - rest) * 100:.1f} cm, 结束位置 {np.round(d.xpos[obj], 3).tolist()}, "
              f"{'被抓住过' if max(z) - rest > 0.02 else '没被抓起'}")

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
                    txt = "contact" if in_contact[t] else ""
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
