"""
一次性: Unitree 官方 URDF (G1 29 自由度 + Inspire 五指手) -> MuJoCo MJCF, 输出 models/g1_inspire/ (连同 meshes 一起提交)

  git clone --depth 1 --filter=blob:none --sparse https://github.com/unitreerobotics/unitree_ros.git
  cd unitree_ros && git sparse-checkout set robots/g1_description
  python tools/urdf_to_mjcf.py --urdf unitree_ros/robots/g1_description/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf

输出:
  models/g1_inspire/g1_inspire.xml   机器人 (freejoint 骨盆 + position 执行器 + mimic 等式约束)
  models/g1_inspire/scene.xml        include 机器人 + 地面 + 灯光 (同 menagerie 的 scene_with_hands.xml)
  models/g1_inspire/meshes/          用到的 STL
  models/g1_inspire/preview_open.png, preview_closed.png   右手张开 / 握拳 (hoi_to_mujoco.py 里 inspire 的 f_open / f_closed)

做法:
  1. URDF 原有的 <mujoco> 换成 <compiler meshdir="meshes" discardvisual="false" balanceinertia="true"
     strippath="true" fusestatic="false"/> (URDF 里的文件名已经带 "meshes/", strippath="false" 会变成 meshes/meshes/),
     MjModel.from_xml_path 加载, mj_saveLastXML 存成 MJCF
  2. <mimic> -> 每个跟随关节一个 <joint joint1=跟随 joint2=主动 polycoef="offset multiplier 0 0 0"/>, offset / multiplier 从 URDF 读
     (MuJoCo 3.17 的 URDF 导入其实已经会生成不带名字的 mimic 等式约束; 删掉它们, 只留这里带名字的一份, 避免重复)
  3. 骨盆加 freejoint; 每个非跟随关节一个 position 执行器 (名字 = 关节名):
     身体/手臂: 同 menagerie G1 (kp 500, dampratio 1, armature 0.01, frictionloss 0.3)
     手指: forcerange = 额定力 x 关节到指尖的距离 (RH56DFQ: 拇指 6 N, 四指 4 N; 经销商公布的规格), kp = forcerange / 0.15 rad
  4. URDF 的 <collision> 保留为碰撞体 (group 3), <visual> 为显示 (group 2, 不碰撞);
     机器人碰撞体 conaffinity=0: 机器人各部件之间不互相碰 (手指网格之间本来就有重叠), 但能碰物体 / 桌子 / 地面
"""
import argparse, os, re, shutil, tempfile
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "models", "g1_inspire")
BASE_Z = 0.793
F_THUMB, F_FINGER = 6.0, 4.0                    # N, RH56DFQ 拇指 / 四指最大握力
FINGER_ERR = 0.15                               # rad: 位置误差这么大时达到 forcerange
TIPS = {                                        # 主动关节 -> 这根手指最末端的 body (算指尖距离)
    "thumb_proximal_yaw_joint": "thumb_distal", "thumb_proximal_pitch_joint": "thumb_distal",
    "index_proximal_joint": "index_intermediate", "middle_proximal_joint": "middle_intermediate",
    "ring_proximal_joint": "ring_intermediate", "pinky_proximal_joint": "pinky_intermediate"}


def load_urdf(urdf):
    """换掉 URDF 里的 <mujoco> 编译选项, 加载, 存成 MJCF 文本"""
    u = open(urdf).read()
    opt = ('<mujoco><compiler meshdir="meshes" discardvisual="false" balanceinertia="true" '
           'strippath="true" fusestatic="false"/></mujoco>')
    u = re.sub(r"<mujoco>.*?</mujoco>", opt, u, count=1, flags=re.S) if "<mujoco>" in u else \
        re.sub(r"(<robot[^>]*>)", r"\1" + opt, u, count=1)
    tmp = os.path.join(os.path.dirname(os.path.abspath(urdf)), "_tmp_to_mjcf.urdf")
    open(tmp, "w").write(u)
    try:
        m = mujoco.MjModel.from_xml_path(tmp)
    finally:
        os.remove(tmp)
    path = os.path.join(tempfile.mkdtemp(), "raw.xml")
    mujoco.mj_saveLastXML(path, m)
    return m, path


def mimics(urdf):
    """URDF 的 <mimic>: {跟随关节: (主动关节, multiplier, offset)}"""
    r = ET.parse(urdf).getroot()
    out = {}
    for j in r.findall("joint"):
        mm = j.find("mimic")
        if mm is not None:
            out[j.get("name")] = (mm.get("joint"), float(mm.get("multiplier", 1)), float(mm.get("offset", 0)))
    return out


def tip_distance(m, joint, tip_body):
    """张开时, 关节轴到指尖 (末端 body 碰撞网格离关节最远的顶点) 的距离"""
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    j = m.joint(joint)
    anchor = d.xanchor[j.id]
    axis = d.xaxis[j.id]
    b = m.body(tip_body).id
    far = 0.0
    for g in np.where(m.geom_bodyid == b)[0]:
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        w = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T - anchor
        w -= np.outer(w @ axis, axis)                  # 到关节轴的垂直距离
        far = max(far, np.linalg.norm(w, axis=1).max())
    return far


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(os.path.join(out, "meshes"), exist_ok=True)

    m, raw = load_urdf(args.urdf)
    mim = mimics(args.urdf)
    tree = ET.parse(raw)
    root = tree.getroot()
    root.set("model", "g1_inspire")

    # 网格: 只拷用到的
    src_dir = os.path.join(os.path.dirname(os.path.abspath(args.urdf)), "meshes")
    for mesh in root.iter("mesh"):
        shutil.copy(os.path.join(src_dir, mesh.get("file")), os.path.join(out, "meshes", mesh.get("file")))

    opt = ET.Element("option", integrator="implicitfast")
    root.insert(1, opt)

    # 骨盆 freejoint
    world = root.find("worldbody")
    pelvis = world.find("body[@name='pelvis']")
    pelvis.set("pos", f"0 0 {BASE_Z}")
    pelvis.insert(1, ET.Element("freejoint", name="floating_base_joint"))

    # 几何体: visual (导入时 contype=0) -> group 2; collision -> group 3, conaffinity 0
    for g in world.iter("geom"):
        if g.get("contype") == "0":
            g.set("group", "2")
        else:
            g.set("group", "3"); g.set("contype", "1"); g.set("conaffinity", "0")

    # 关节阻尼 / 惯量
    for j in world.iter("joint"):
        n = j.get("name")
        hand = n[:2] in ("L_", "R_")
        j.set("armature", "0.001" if hand else "0.01")
        j.set("frictionloss", "0.01" if hand else "0.3")
        if hand:
            j.set("damping", "0.01")

    # mimic -> 等式约束 (先删掉导入时自动生成的)
    for old in root.findall("equality"):
        root.remove(old)
    eq = ET.SubElement(root, "equality")
    for follower, (leader, mult, off) in mim.items():
        ET.SubElement(eq, "joint", name=f"mimic_{follower}", joint1=follower, joint2=leader,
                      polycoef=f"{off:g} {mult:g} 0 0 0", solref="0.005 1")

    # 执行器
    act = ET.SubElement(root, "actuator")
    info = []
    for j in world.iter("joint"):
        n = j.get("name")
        if n in mim:
            continue
        lo, hi = map(float, j.get("range").split())
        side_key = n[2:] if n[:2] in ("L_", "R_") else None
        if side_key in TIPS:
            tip = n[:2] + TIPS[side_key]
            L = tip_distance(m, n, tip)
            tau = (F_THUMB if "thumb" in n else F_FINGER) * L
            kp = tau / FINGER_ERR
            ET.SubElement(act, "position", name=n, joint=n, kp=f"{kp:.4g}", dampratio="1",
                          ctrlrange=f"{lo:g} {hi:g}", forcerange=f"{-tau:.4g} {tau:.4g}")
            info.append((n, L, tau, kp))
        else:
            ET.SubElement(act, "position", name=n, joint=n, kp="500", dampratio="1", ctrlrange=f"{lo:g} {hi:g}")

    ET.indent(tree, "  ")
    tree.write(os.path.join(out, "g1_inspire.xml"))
    open(os.path.join(out, "scene.xml"), "w").write(SCENE)

    mm = mujoco.MjModel.from_xml_path(os.path.join(out, "scene.xml"))      # 检查能加载
    print(f"saved {out}: nq {mm.nq}, nu {mm.nu}, neq {mm.neq}, ngeom {mm.ngeom}")
    for n, L, tau, kp in info:
        if n.startswith("R_"):
            print(f"  {n:30s} 指尖距离 {L * 100:.1f} cm  forcerange ±{tau:.3f} N·m  kp {kp:.2f}")
    render_previews(out)


def render_previews(out):
    """右手张开 / 握拳各一张 (手臂在 hoi_to_mujoco.q0)"""
    import sys
    import imageio
    sys.path.insert(0, os.path.join(HERE, ".."))
    import hoi_to_mujoco as H
    H.use_hand("inspire")
    m = mujoco.MjModel.from_xml_path(os.path.join(out, "scene.xml"))
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 0.45, 200, -30
    for name, f in [("open", H.F_OPEN), ("closed", H.F_CLOSED)]:
        mujoco.mj_resetData(m, d)
        d.qpos[:7] = H.BASE
        d.qpos[[m.joint(n).qposadr[0] for n in H.ARM]] = H.q0
        H.set_fingers(m, d, f)
        mujoco.mj_forward(m, d)
        cam.lookat[:] = d.xpos[m.body("R_hand_base_link").id]
        r.update_scene(d, cam)
        imageio.imwrite(os.path.join(out, f"preview_{name}.png"), r.render())


SCENE = """<mujoco model="g1_inspire scene">
  <include file="g1_inspire.xml"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0.9 0.9 0.9"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="140" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true" diffuse="0.5 0.5 0.5"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
"""

if __name__ == "__main__":
    main()
