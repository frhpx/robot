import mujoco
import mujoco.viewer
import numpy as np
import time

m = mujoco.MjModel.from_xml_path(r"C:\Users\30691\mujoco_menagerie\unitree_g1\scene_with_hands.xml")
d = mujoco.MjData(m)
dk = mujoco.MjData(m)          # 专门给 IK 算 FK 用, 不影响仿真

ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
       "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
qa = np.array([m.joint(n).qposadr[0] for n in ARM])
va = np.array([m.joint(n).dofadr[0] for n in ARM])
lo, hi = np.array([m.joint(n).range for n in ARM]).T
ee = m.body("right_wrist_yaw_link").id
BASE = [0, 0, 0.79, 1, 0, 0, 0]


def ik(p_t, quat_t, q_prev, q_rest, iters=30, lam=0.05, w_prev=1e-3, w_rest=0.05):
    q = q_prev.copy()
    Jp, Jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(iters):
        dk.qpos[qa] = q
        mujoco.mj_kinematics(m, dk); mujoco.mj_comPos(m, dk)            # FK
        qd, w = np.zeros(4), np.zeros(3)
        mujoco.mju_negQuat(qd, dk.xquat[ee]); mujoco.mju_mulQuat(qd, quat_t, qd)
        mujoco.mju_quat2Vel(w, qd, 1.0)
        e = np.r_[p_t - dk.xpos[ee], w]                                 # 6 维误差
        if np.linalg.norm(e) < 1e-5:
            break
        mujoco.mj_jacBody(m, dk, Jp, Jr, ee)
        J = np.vstack([Jp, Jr])[:, va]                                  # 6x7 雅可比
        dq = np.linalg.solve(J.T @ J + (lam**2 + w_prev) * np.eye(7),   # DLS + 离上一帧近
                             J.T @ e - w_prev * (q - q_prev))
        N = np.eye(7) - np.linalg.pinv(J) @ J                           # 零空间拉向参考姿势
        q = np.clip(q + dq - N @ (w_rest * (q - q_rest)), lo, hi)
    return q


# ---------- 1. 离线算好整条 8 字轨迹的关节角 ----------
dk.qpos[:7] = BASE
q = np.zeros(7); q[0], q[3] = -0.4, 0.9
dk.qpos[qa] = q; mujoco.mj_kinematics(m, dk)
quat0, c = dk.xquat[ee].copy(), dk.xpos[ee] + [0, -0.05, 0.13]   # 中心: 上方 13 cm, 往外 5 cm (避免肩撞躯干)
t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
targets = c + np.c_[0.04 * np.sin(t), 0.08 * np.sin(t), 0.06 * np.sin(2 * t)]   # 3D 8 字

q = ik(targets[0], quat0, q, q, iters=300, w_prev=0, w_rest=0)          # 先到轨迹第一个点
q_rest, traj = q.copy(), []
for p in targets:
    q = ik(p, quat0, q, q_rest)
    traj.append(q)
traj = np.array(traj)

# ---------- 2. 仿真: 位置执行器跟踪关节轨迹 ----------
T_LAP = 4.0                                                             # 4 秒走一遍 8 字
adof = m.jnt_dofadr[m.actuator_trnid[:, 0]]                             # 每个执行器对应的自由度
kp = m.actuator_gainprm[:, 0]
aid = [m.actuator(n).id for n in ARM]
with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        k = int(d.time / T_LAP * len(traj)) % len(traj)
        target = np.zeros(m.nu)
        target[aid] = traj[k]
        d.ctrl[:] = target + d.qfrc_bias[adof] / kp                    # 重力补偿

        d.qpos[:7] = BASE
        d.qvel[:6] = 0

        mujoco.mj_step(m, d)
        viewer.sync()
        time.sleep(0.002)