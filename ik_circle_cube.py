import mujoco
import mujoco.viewer
import numpy as np
import time

m = mujoco.MjModel.from_xml_path(r"C:\Users\30691\mujoco_menagerie\unitree_g1\scene_with_cube.xml")
d = mujoco.MjData(m)

# 右臂 7 个关节: 在 qpos 里的位置, 在 qvel(=雅可比的列) 里的位置
ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
       "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
qa = [m.joint(n).qposadr[0] for n in ARM]
va = [m.joint(n).dofadr[0] for n in ARM]
hand = m.body("right_wrist_yaw_link").id
cube = m.body("cube").mocapid[0]               # 方块的 mocap 编号

# 手指合拢 (握住方块的姿势)
GRASP = {"right_hand_thumb_1_joint": 0.3, "right_hand_thumb_2_joint": -0.4,
         "right_hand_index_0_joint": 0.5, "right_hand_index_1_joint": 0.5,
         "right_hand_middle_0_joint": 0.5, "right_hand_middle_1_joint": 0.5}
for name, angle in GRASP.items():
    d.qpos[m.joint(name).qposadr[0]] = angle

# 方块在手腕坐标系里的位置: 往前 11 cm, 往拇指方向 3.5 cm
CUBE_OFFSET = np.array([0.11, 0.035, 0.0])

# 起始姿势: 手臂抬起, 肘弯曲
q0 = np.array([-0.4, 0, 0, 0.9, 0, 0, 0])
d.qpos[qa] = q0
mujoco.mj_forward(m, d)
center = d.xpos[hand] + [0, -0.05, 0.13]       # 圆心: 起始手腕上方 13 cm, 往外 5 cm
R = 0.08                                       # 半径 8 cm

with mujoco.viewer.launch_passive(m, d) as viewer:
    t = 0
    while viewer.is_running():
        # 1. 这一帧手腕要去的点 (4 秒一圈)
        a = 2 * np.pi * t / 4
        target = center + R * np.array([0, np.cos(a), np.sin(a)])

        # 2. 先往起始姿势拉一点 (防止画久了手臂姿势越漂越远)
        d.qpos[qa] += 0.05 * (q0 - d.qpos[qa])

        # 3. IK: 重复 "FK -> 误差 -> 雅可比 -> 改一点关节角"
        for _ in range(10):
            mujoco.mj_forward(m, d)                          # FK: 手腕现在在哪
            err = target - d.xpos[hand]                      # 误差 (3 个数)
            Jp = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, Jp, None, hand)          # 雅可比
            J = Jp[:, va]                                    # 只要右臂 7 列 -> 3x7
            dq = J.T @ np.linalg.solve(J @ J.T + 0.05**2 * np.eye(3), err)   # 阻尼最小二乘
            d.qpos[qa] += dq                                 # 改一点关节角

        # 4. 方块跟着手走: 方块位置 = 手腕位置 + 手腕朝向 x 偏移
        mujoco.mj_forward(m, d)
        Rh = d.xmat[hand].reshape(3, 3)                      # 手腕的旋转矩阵 (3x3)
        d.mocap_pos[cube] = d.xpos[hand] + Rh @ CUBE_OFFSET
        d.mocap_quat[cube] = d.xquat[hand]                   # 方块朝向 = 手腕朝向

        # 5. 显示
        mujoco.mj_forward(m, d)
        viewer.sync()
        time.sleep(0.01)
        t += 0.01