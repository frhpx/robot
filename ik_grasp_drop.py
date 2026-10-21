import mujoco
import mujoco.viewer
import numpy as np
import time

m = mujoco.MjModel.from_xml_path(r"C:\Users\30691\mujoco_menagerie\unitree_g1\scene_grasp.xml")
d = mujoco.MjData(m)        # 物理仿真用
dk = mujoco.MjData(m)       # IK 试算用

# ---------- 编号 ----------
ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
       "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
FINGERS = ["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
           "right_hand_index_0_joint", "right_hand_index_1_joint",
           "right_hand_middle_0_joint", "right_hand_middle_1_joint"]
F_CLOSED = np.array([0, 0.3, -0.8, 1.2, 1.2, 1.2, 1.2])       # 手指握紧时的目标角度

qa = [m.joint(n).qposadr[0] for n in ARM]
va = [m.joint(n).dofadr[0] for n in ARM]
fq = [m.joint(n).qposadr[0] for n in FINGERS]
arm_act = [m.actuator(n).id for n in ARM]
finger_act = [m.actuator(n).id for n in FINGERS]
hand = m.body("right_wrist_yaw_link").id
cube = m.body("cube").id
cube_q = m.joint("cube_joint").qposadr[0]
cube_v = m.joint("cube_joint").dofadr[0]
weld = m.equality("grasp").id
thumb = {m.body(f"right_hand_thumb_{i}_link").id for i in range(3)}
finger = {m.body(f"right_hand_{f}_{i}_link").id for f in ("index", "middle") for i in range(2)}

adof = m.jnt_dofadr[m.actuator_trnid[:, 0]]     # 每个电机对应的自由度 (重力补偿用)
kp = m.actuator_gainprm[:, 0]
BASE = [0, 0, 0.793, 1, 0, 0, 0]
GRASP_OFFSET = np.array([0.12, 0.05, 0.0])      # 抓取点: 拇指和手指之间, 手腕坐标系


def grasp_point(data):
    return data.xpos[hand] + data.xmat[hand].reshape(3, 3) @ GRASP_OFFSET


def ik(target):
    """位置 IK (在 dk 上算): 让抓取点到 target"""
    dk.qpos[qa] += 0.05 * (q0 - dk.qpos[qa])
    for _ in range(10):
        mujoco.mj_kinematics(m, dk); mujoco.mj_comPos(m, dk)
        p = grasp_point(dk)
        Jp = np.zeros((3, m.nv))
        mujoco.mj_jac(m, dk, Jp, None, p, hand)
        J = Jp[:, va]
        dk.qpos[qa] += J.T @ np.linalg.solve(J @ J.T + 0.05**2 * np.eye(3), target - p)


def touching():
    """方块是否同时碰到拇指和 (食指或中指)"""
    t = f = False
    for c in d.contact[:d.ncon]:
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        other = b2 if b1 == cube else b1 if b2 == cube else None
        t |= other in thumb
        f |= other in finger
    return t and f


def set_weld():
    """打开焊接: 把方块固定在手上, 保持当前的相对位置"""
    R = d.xmat[hand].reshape(3, 3)
    m.eq_data[weld, 3:6] = R.T @ (d.xpos[cube] - d.xpos[hand])      # 方块在手腕坐标系里的位置
    qinv = np.zeros(4); rel = np.zeros(4)
    mujoco.mju_negQuat(qinv, d.xquat[hand]); mujoco.mju_mulQuat(rel, qinv, d.xquat[cube])
    m.eq_data[weld, 6:10] = rel                                      # 方块相对手腕的朝向
    d.eq_active[weld] = 1


def smooth(s):
    return 10 * s**3 - 15 * s**4 + 6 * s**5


# ---------- 起始状态和动作序列 ----------
q0 = np.array([-0.4, 0, 0, 0.9, 0, 0, 0])
A = np.array([0.33, -0.22, 0.805])     # 抓取点目标: 方块中心上方 1 cm
UP = np.array([0, 0, 0.15])            # 举起 15 cm
dk.qpos[:7] = BASE; dk.qpos[qa] = q0
mujoco.mj_kinematics(m, dk)
START = grasp_point(dk)

# (时间, 抓取点起点, 抓取点终点, 手指起点, 手指终点)  手指: 0 张开, 1 握紧
PHASES = [
    (1.0, START,      START + UP, 0, 0),   # 1. 先抬高 (避免横扫碰倒方块)
    (1.2, START + UP, A + UP,     0, 0),   # 2. 移到方块正上方
    (1.5, A + UP,     A,          0, 0),   # 3. 竖直下降
    (0.8, A,          A,          0, 1),   # 4. 合拢手指 -> 碰到就抓住
    (1.5, A,          A + UP,     1, 1),   # 5. 举起
    (0.5, A + UP,     A + UP,     1, 1),   # 6. 停一下
    (0.3, A + UP,     A + UP,     1, 0),   # 7. 松手 -> 方块自由落体
    (2.0, A + UP,     A + UP,     0, 0),   # 8. 看它落下
]


def reset():
    mujoco.mj_resetData(m, d)
    d.qpos[:7] = BASE; d.qpos[qa] = q0
    d.qpos[cube_q:cube_q + 7] = [0.33, -0.22, 0.795, 1, 0, 0, 0]
    d.eq_active[weld] = 0
    dk.qpos[:] = d.qpos
    mujoco.mj_forward(m, d)


with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        reset()
        held_fingers = None
        for i, (T, p_start, p_end, f_start, f_end) in enumerate(PHASES):
            if i == 6:                                   # 松手: 关掉焊接, 方块交给物理
                d.eq_active[weld] = 0
                held_fingers = None
            n = int(T / m.opt.timestep)
            for k in range(n):
                s = smooth(k / n)

                # 1. 手臂: IK 算出关节目标
                ik(p_start + s * (p_end - p_start))

                # 2. 手指: 张开/握紧之间插值; 抓住后保持当时的角度
                f_target = (f_start + s * (f_end - f_start)) * F_CLOSED
                if held_fingers is not None:
                    f_target = held_fingers

                # 3. 合拢阶段: 拇指和手指都碰到方块 -> 打开焊接
                if i == 3 and held_fingers is None and touching():
                    set_weld()
                    held_fingers = d.qpos[fq].copy()

                # 4. 电机命令 + 重力补偿, 骨盆固定
                target = np.zeros(m.nu)
                target[arm_act] = dk.qpos[qa]
                target[finger_act] = f_target
                d.ctrl[:] = target + d.qfrc_bias[adof] / kp
                d.qpos[:7] = BASE; d.qvel[:6] = 0

                mujoco.mj_step(m, d)
                if k % 5 == 0:                           # 每 5 步刷新一次画面 (100 Hz)
                    viewer.sync()
                    time.sleep(0.01)
                if not viewer.is_running():
                    break
