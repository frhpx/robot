import mujoco
import mujoco.viewer
import numpy as np
import time

m = mujoco.MjModel.from_xml_path(r"C:\Users\30691\mujoco_menagerie\unitree_g1\scene_pick.xml")
d = mujoco.MjData(m)

# ---------- 编号 ----------
ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
       "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
qa = [m.joint(n).qposadr[0] for n in ARM]
va = [m.joint(n).dofadr[0] for n in ARM]
hand = m.body("right_wrist_yaw_link").id
cube = m.body("cube").mocapid[0]

# 手指: 张开 = 0, 握紧 = 下面的角度
FINGERS = {"right_hand_thumb_1_joint": 0.3, "right_hand_thumb_2_joint": -0.4,
           "right_hand_index_0_joint": 0.5, "right_hand_index_1_joint": 0.5,
           "right_hand_middle_0_joint": 0.5, "right_hand_middle_1_joint": 0.5}
fa = [m.joint(n).qposadr[0] for n in FINGERS]
f_closed = np.array(list(FINGERS.values()))

# 抓取点: 拇指和四指之间, 在手腕坐标系里往前 11 cm, 往拇指方向 3.5 cm
GRASP_OFFSET = np.array([0.11, 0.035, 0.0])


def grasp_point():
    """抓取点在世界坐标系里的位置"""
    return d.xpos[hand] + d.xmat[hand].reshape(3, 3) @ GRASP_OFFSET


# ---------- 位置 ----------
A = np.array([0.33, -0.12, 0.76])     # 方块起始位置 (桌上)
B = np.array([0.33, -0.25, 0.76])     # 放下位置 (绿色标记)
UP = np.array([0, 0, 0.12])           # 举起高度 12 cm

q0 = np.array([-0.4, 0, 0, 0.9, 0, 0, 0])
d.qpos[qa] = q0
mujoco.mj_forward(m, d)
START = grasp_point()

# ---------- 动作序列: (持续时间, 抓取点起点, 抓取点终点, 手指起点, 手指终点) ----------
# 手指: 0 = 张开, 1 = 握紧
PHASES = [
    (1.5, START, A + UP, 0, 0),    # 1. 移到方块上方
    (1.0, A + UP, A,     0, 0),    # 2. 下降
    (0.8, A,      A,     0, 1),    # 3. 合拢手指 (抓取)
    (1.2, A,      A + UP, 1, 1),   # 4. 举起
    (1.5, A + UP, B + UP, 1, 1),   # 5. 平移到放下位置上方
    (1.0, B + UP, B,     1, 1),    # 6. 下降
    (0.8, B,      B,     1, 0),    # 7. 张开手指 (放下)
    (1.0, B,      B + UP, 0, 0),   # 8. 手抬起离开
]


def smooth(s):
    """五次多项式: 0->1, 起点和终点的速度、加速度都是 0"""
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def ik(target):
    """位置 IK: 让抓取点到 target (阻尼最小二乘)"""
    d.qpos[qa] += 0.05 * (q0 - d.qpos[qa])                 # 往起始姿势拉一点
    for _ in range(10):
        mujoco.mj_forward(m, d)
        p = grasp_point()
        err = target - p
        Jp = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, Jp, None, p, hand)             # 抓取点的雅可比
        J = Jp[:, va]
        dq = J.T @ np.linalg.solve(J @ J.T + 0.05**2 * np.eye(3), err)
        d.qpos[qa] += dq


with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        # 每一轮: 方块放回 A, 手回到起始姿势
        d.mocap_pos[cube] = A
        d.qpos[qa] = q0
        holding = False

        for T, p_start, p_end, f_start, f_end in PHASES:
            for k in range(int(T / 0.01)):
                s = smooth(k / (T / 0.01))                   # 这一阶段走了多少 (0~1)

                # 1. 手: IK 让抓取点到插值位置
                ik(p_start + s * (p_end - p_start))

                # 2. 手指: 在张开和握紧之间插值
                closeness = f_start + s * (f_end - f_start)
                d.qpos[fa] = closeness * f_closed

                # 3. 方块: 手指握紧时跟着抓取点走, 张开时留在原地
                if closeness > 0.9:
                    holding = True
                elif closeness < 0.5:
                    holding = False
                mujoco.mj_forward(m, d)
                if holding:
                    d.mocap_pos[cube] = grasp_point()

                # 4. 显示
                mujoco.mj_forward(m, d)
                viewer.sync()
                time.sleep(0.01)
                if not viewer.is_running():
                    break
        time.sleep(0.5)
