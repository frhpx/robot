import mujoco
import mujoco.viewer
import numpy as np
import time


m=mujoco.MjModel.from_xml_path(r"C:\Users\30691\mujoco_menagerie\unitree_g1\scene_with_hands.xml")
d=mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        
        d.actuator('right_shoulder_pitch_joint').ctrl = np.sin(d.time)
        
        d.qpos[:7]= [0,0,0.79,1,0,0,0]
        d.qvel[:6] = 0
        
        mujoco.mj_step(m, d)
        viewer.sync()
        time.sleep(0.002)