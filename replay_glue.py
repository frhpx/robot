"""
grasp_glue.py 的输出 -> MuJoCo G1 (Inspire 手) 复现: 抓取点跟着人手, 接触段里手和物体粘在一起一起动, 不掉落, 动作连续

  python replay_glue.py --object assets/mouse_pts                                  # 开窗口
  python replay_glue.py --object assets/mouse_pts --video out.mp4 --src-video data/mouse.mp4   # 左原视频 右机器人

输入: <object>/glue.npz (物体位姿 + 手 21 点, 桌面系), scene.json / meta.json (桌面系 -> 世界系: 同 hoi_to_mujoco.py)
流程:
  1. 桌面系 -> 世界系: hoi_to_mujoco.scene_shift (物体落在 OBJ_XY, 桌面 = TABLE_TOP), 只平移
  2. 机器人抓取点 (手张开时拇指尖和食指/中指尖的中点) 的目标: 接触前 = 人手 拇指尖 和 (食指 + 中指尖)/2 的中点;
     接触段里 = 物体坐标系里的固定点 (--grasp plan: grasp_plan.py 规划的抓取; human: 人手抓的位置), 跟着物体轨迹刚性运动;
     接触段前后 BLEND 帧内, 从人手原始轨迹平滑过渡到粘合轨迹 (smoothstep), 不跳变;
     接触前离物体水平越远, 目标越高 (从上方落下, 不横扫碰倒)
  3. 手腕朝向 = 物体朝向 x 规划的抓取朝向 (物体系): 接触前不动, 接触段里跟着物体一起转
     手指合拢目标 = 规划的每根手指刚好贴上物体表面的角度 (不是握成拳)
  4. 接触段开始: 抓取点到位 (< ARRIVE) 后手指用 smoothstep 在 CLOSE 帧内合拢到规划角度; 规划里贴着但物理上还没碰到的手指
     每帧再合一点 (挤压, 最多 SQUEEZE 帧); 碰到物体的手指停在碰到时的角度, 不再往里推 (否则把物体推走);
     规划的手指都碰到 / 都在 2 mm 以内 / 物体开始被推动 (> 2 mm) -> 打开焊接 (粘合),
     按当时的相对位姿焊, 不跳; 手指没两边都碰到也焊 (保证不掉); 焊着时关掉物体和机器人的碰撞 (还和桌子碰), 松手后手张开、抬起了再恢复. 到位 = 离目标 < ARRIVE, 或压住了靠近不了 (STALL). 焊上之后手的目标 = 物体粘合轨迹 x 焊上时的手-物体相对位姿,
     所以物体 (不是手) 精确走人的轨迹. 焊上时按实际手-物体位姿重新找每根手指刚好贴上表面的角度, 焊着期间保持. 接触段结束 -> 关焊接, 手臂原地不动、手指 CLOSE 帧内张开, 再先抬起后离开
  5. IK / 电机 / 重力补偿 同 hoi_to_mujoco.py; 关节速度限制保证手臂连续;
     另加积分修正: 位置电机带负载有几 mm 稳态误差, 不修的话手到不了规划位姿, 拇指够不着, 手指一挤把物体推走
检查 (打印): 焊接是否覆盖整个接触段、物体相对粘合轨迹的误差、手臂 / 手指最大速度和加速度、松手时物体下落高度
"""
import argparse, json, os, time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

import hoi_to_mujoco as H
from track_object import smooth_poses
import grasp_plan as GP

BLEND = 15                 # 接触段前后的过渡帧数
CLOSE = 10                 # 手指合拢 / 张开用的帧数
ARRIVE = 0.005             # 抓取点离目标 < 5 mm 算到位
SQUEEZE = 10               # 合拢到规划角度后, 还没碰到物体的手指每帧再合 SQ_STEP, 最多这么多帧
SQ_STEP = 0.04             # rad / 帧
K_I = 0.3                  # 手臂稳态误差的积分修正: 每帧把 (目标 - 实际抓取点) 的 K_I 加进 IK 目标
I_MAX = 0.02               # 修正量上限 (m)
WAIT = 20                  # 接触开始后最多等这么多帧到位, 之后不等了直接合拢
STALL = 0.001              # 离目标 < 6 cm 且一帧靠近不到 1 mm (手指压在物体 / 桌面上了) 也算到位
HOVER = 0.08               # 接触前离物体水平距离 > 3 cm 时, 抓取点至少比物体中心高这么多
PAD = 30
ROBOT_SMOOTH = 9           # 机器人跟随的物体位姿 / 抓取点目标的平滑窗口 (帧)


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def human_grasp_point(kp):
    """人手 拇指尖 和 (食指尖 + 中指尖)/2 的中点 (同 Inspire 抓取点的定义)"""
    return (kp[..., 4, :] + (kp[..., 8, :] + kp[..., 12, :]) / 2) / 2


def build_targets(g, shift, a, b, g_plan=None):
    """世界系: 物体位姿, 抓取点目标, 每帧接触标记 / 合拢权重, 手腕朝向的旋转 (相对接触开始那帧)"""
    R, t = g["obj_R"][a:b], g["obj_t"][a:b] + shift
    raw = human_grasp_point(g["kp"][a:b]) + shift       # hand_track.py 已经平滑过
    contact = g["contact"][a:b]
    n = b - a
    glued = raw.copy()
    rot = np.tile(np.eye(3), (n, 1, 1))                  # 手腕要跟着转的旋转 (相对段首)
    w = np.zeros(n)                                       # 1 = 用粘合轨迹, 0 = 用人手原始轨迹
    for s, e in g["segs"]:
        s, e = s - a, e - a
        if e <= 0 or s >= n:
            continue
        s0, e0 = max(s, 0), min(e, n)
        g_obj = np.median([R[i].T @ (raw[i] - t[i]) for i in range(s0, e0)], 0)   # 抓取点在物体坐标系里 (粘合后段内是常数)
        if g_plan is not None:                            # 机器人自己的抓取 (grasp_plan.py): 人手常握在边上, Inspire 照抄会压到边、手指撞桌子
            g_obj = g_plan
        for i in range(n):                                # 段前段后用物体的位姿延伸, 过渡时两头都是连续的
            j = min(max(i, s0), e0 - 1)
            if s0 - BLEND <= i < e0 + BLEND:
                glued[i] = R[j] @ g_obj + t[j] if s0 <= i < e0 else R[i] @ g_obj + t[i]
        for i in range(max(0, s0 - BLEND), min(n, e0 + BLEND)):
            w[i] = max(w[i], smoothstep((i - (s0 - BLEND)) / BLEND) if i < s0 else
                       1.0 if i < e0 else smoothstep(1 - (i - e0 + 1) / BLEND))
        for i in range(s0, n):
            rot[i] = R[min(i, e0 - 1)] @ R[s0].T
    target = (1 - w[:, None]) * raw + w[:, None] * glued
    return R, t, target, contact, rot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", required=True, help="有 glue.npz (grasp_glue.py 的输出)")
    ap.add_argument("--seg", type=int, default=0, help="复现第几个接触段")
    ap.add_argument("--video", default=None)
    ap.add_argument("--src-video", default=None)
    ap.add_argument("--menagerie", default=r"C:\Users\30691\mujoco_menagerie", help="只有 --hand dex3 需要")
    ap.add_argument("--hand", default="inspire", choices=list(H.HANDS))
    ap.add_argument("--grasp", default="plan", choices=["plan", "human"],
                    help="plan = grasp_plan.py 规划 (手指真的贴在物体表面); human = 照抄人手 拇指和食指/中指中点在物体上的位置")
    ap.add_argument("--max-vel", type=float, default=6.0, help="手臂关节最大速度 (rad/s); 人快速抬起时 3 跟不上")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.object, "meta.json")))
    g = dict(np.load(os.path.join(args.object, "glue.npz")))
    fps = float(g["fps"])
    n_all = len(g["obj_t"])
    s, e = g["segs"][args.seg]
    a, b = max(0, s - PAD), min(n_all, e + PAD)
    shift = H.scene_shift(meta)
    R, t, _, contact, _ = build_targets(g, shift, a, b)
    R, t = smooth_poses(R, t, ROBOT_SMOOTH)               # 机器人跟的轨迹再平滑一次: 物体朝向的抖动会变成手腕关节的速度尖峰
    n = b - a
    print(f"接触段 {args.seg}: 帧 {s}-{e}, 回放 {a}-{b}; 物体最高 {(t[:, 2].max() - t[0, 2]) * 100:.1f} cm")

    H.use_hand(args.hand)
    H.OBJ_POS = t[0].copy()
    m = H.build_model(args.menagerie, args.object)
    if H.GRASP_OFFSET is None:
        H.GRASP_OFFSET = H.calibrate_grasp_offset(m)
    d, dk = mujoco.MjData(m), mujoco.MjData(m)
    ARM, FINGERS, F_OPEN, F_CLOSED = H.ARM, H.FINGERS, H.F_OPEN, H.F_CLOSED
    qa = [m.joint(j).qposadr[0] for j in ARM]
    va = [m.joint(j).dofadr[0] for j in ARM]
    fq = [m.joint(j).qposadr[0] for j in FINGERS]
    arm_act = [m.actuator(j).id for j in ARM]
    finger_act = [m.actuator(j).id for j in FINGERS]
    adof = m.jnt_dofadr[m.actuator_trnid[:, 0]]
    kp = m.actuator_gainprm[:, 0]
    hand = m.body(H.HAND["wrist"]).id
    obj = m.body("obj").id
    obj_q = m.joint("obj_joint").qposadr[0]
    weld = m.equality("grasp").id
    thumb = {m.body(x).id for x in H.HAND["thumb"]}
    others = {m.body(x).id for x in H.HAND["others"]}
    n_sub = max(1, round(1 / (fps * m.opt.timestep)))
    lo, hi = m.jnt_range[[m.joint(j).id for j in ARM]].T
    obj_geoms = np.where(m.geom_bodyid == obj)[0]
    aff0 = m.geom_conaffinity[obj_geoms].copy()

    def glue_collisions(on):
        """焊着时物体不和机器人碰 (机器人 contype 1 / conaffinity 0): 物体 conaffinity 改成 2;
           物体 contype 还是 1, 照样和桌子 / 地面碰. 否则手指压进物体的接触力会和焊接、手臂打架"""
        m.geom_conaffinity[obj_geoms] = 2 if on else aff0
    q0 = H.q0

    def grasp_point(data):
        return data.xpos[hand] + data.xmat[hand].reshape(3, 3) @ H.GRASP_OFFSET

    q_goal = np.zeros(4)
    max_dq = [np.inf]

    def ik(p_target, quat):
        q_prev = dk.qpos[qa].copy()
        for _ in range(10):
            mujoco.mj_kinematics(m, dk); mujoco.mj_comPos(m, dk)
            p = grasp_point(dk)
            Jp, Jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jac(m, dk, Jp, Jr, p, hand)
            err, J = p_target - p, Jp[:, va]
            if quat is not None:
                qc, dr = np.zeros(4), np.zeros(3)
                mujoco.mju_negQuat(qc, dk.xquat[hand]); mujoco.mju_mulQuat(qc, quat, qc)
                mujoco.mju_quat2Vel(dr, qc, 1.0)
                err, J = np.r_[err, H.W_ROT * dr], np.r_[J, H.W_ROT * Jr[:, va]]
            Jinv = J.T @ np.linalg.inv(J @ J.T + 0.05 ** 2 * np.eye(len(err)))
            dk.qpos[qa] += Jinv @ err + (np.eye(7) - Jinv @ J) @ (H.K_NULL * (q0 - dk.qpos[qa]))
        dk.qpos[qa] = np.clip(q_prev + np.clip(dk.qpos[qa] - q_prev, -max_dq[0], max_dq[0]), lo, hi)

    def touching():
        th = fi = False
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            o = b2 if b1 == obj else b1 if b2 == obj else None
            th |= o in thumb; fi |= o in others
        return th, fi

    def touched():
        """物理里正碰着物体的手指 (GROUPS 的下标)"""
        out = set()
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            o = b2 if b1 == obj else b1 if b2 == obj else None
            out |= {k for k, bs in groups.items() if o in bs}
        return out

    def contact_angles(f_now):
        """焊上时: 在当前状态的副本上, 每根规划的手指从张开往握紧扫, 找第一次贴上物体表面的角度 (二分到 < 0.5 mm);
           哪个角度都碰不到的手指取离表面最近的角度. 合拢不是单调靠近表面 (指尖会越过切点), 所以不能用 "间隙大就再合" 的反馈"""
        dd = mujoco.MjData(m)
        dd.qpos[:], dd.qvel[:] = d.qpos, 0
        f = f_now.copy()

        def gap(k, x):
            ff = f.copy(); ff[k] = x
            H.set_fingers(m, dd, ff); mujoco.mj_kinematics(m, dd)
            return GP.finger_gaps(m, dd, {k: groups[k]})[GP.NAMES[k]]
        for k in plan_touch:
            xs = np.linspace(F_OPEN[k], F_CLOSED[k], 41)
            prev, best = xs[0], (np.inf, f[k])
            for x in xs[1:]:
                gx = gap(k, x)
                best = min(best, (gx, x))
                if gx <= 0:
                    lo_, hi_ = prev, x
                    for _ in range(8):
                        mid = (lo_ + hi_) / 2
                        lo_, hi_ = (lo_, mid) if gap(k, mid) <= 0 else (mid, hi_)
                    f[k] = lo_
                    break
                prev = x
            else:                                         # 哪个角度都碰不到: 取离表面最近的角度
                f[k] = best[1]
        return f

    def set_weld():
        Rh = d.xmat[hand].reshape(3, 3)
        m.eq_data[weld, 0:3] = 0
        m.eq_data[weld, 3:6] = Rh.T @ (d.xpos[obj] - d.xpos[hand])
        qinv, rel = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(qinv, d.xquat[hand]); mujoco.mju_mulQuat(rel, qinv, d.xquat[obj])
        m.eq_data[weld, 6:10] = rel
        m.eq_solref[weld] = [2 * m.opt.timestep, 1]       # 最硬的稳定设置: 默认 0.02 s 太软, 加速时物体相对手漂几 mm
        d.eq_active[weld] = 1

    def step(arm_q, finger_q):
        tgt = np.zeros(m.nu)
        tgt[arm_act], tgt[finger_act] = arm_q, finger_q
        for _ in range(n_sub):
            d.ctrl[:] = tgt + d.qfrc_bias[adof] / kp
            d.qpos[:7] = H.BASE; d.qvel[:6] = 0
            mujoco.mj_step(m, d)

    # 抓取朝向: 从 q0 只做位置 IK 到物体上方, 记下手腕朝向 (同 hoi_to_mujoco.py)
    dk.qpos[:7] = H.BASE; dk.qpos[qa] = q0
    for _ in range(50):
        ik(t[s - a] + [0, 0, 0.01], None)
    q_goal[:] = dk.xquat[hand]
    plan = GP.plan_grasp(m, R[s - a], t[s - a], q_goal.copy()) if args.grasp == "plan" else None
    q_rel = plan["q_rel"] if plan else GP.quat_mul(GP.mat2quat(R[s - a].T), q_goal)
    f_grasp = plan["f"] if plan else F_CLOSED
    groups = plan["groups"] if plan else {k: {m.body(x).id for x in v} for k, v in GP.GROUPS.items()}
    _, _, target, _, _ = build_targets(g, shift, a, b, plan["g_obj"] if plan else None)
    plan_touch = plan["touch"] if plan else list(GP.GROUPS)

    def wrist_quat(i):
        """手腕朝向 = 物体朝向 x 抓取时手在物体系里的朝向 (接触前 / 松手后用段首 / 段尾的物体朝向)"""
        return GP.quat_mul(GP.mat2quat(R[min(max(i, s - a), e - a - 1)]), q_rel)

    target = H.smooth(target, ROBOT_SMOOTH)
    # 接触前 / 松手后: 离物体水平越远目标越高 (从上方落下, 从上方离开, 不横扫物体)
    # 松手后先原地停 CLOSE 帧等手指张开
    hold = e - a
    for i in range(hold, min(n, hold + CLOSE)):
        target[i] = target[hold - 1]
    for i in [*range(s - a), *range(min(n, hold + CLOSE), n)]:
        dxy = np.linalg.norm(target[i, :2] - t[i, :2])
        target[i, 2] = max(target[i, 2], t[i, 2] + HOVER * smoothstep((dxy - 0.01) / 0.03))
    for i in range(min(n, hold + CLOSE), min(n, hold + CLOSE + BLEND)):   # 先抬起再走: 高度先到, 水平慢慢过去
        w = smoothstep((i - hold - CLOSE + 1) / BLEND)
        target[i, :2] = (1 - w) * target[hold - 1, :2] + w * target[i, :2]
        target[i, 2] = max(target[i, 2], t[i, 2] + HOVER * w)
    dk.qpos[qa] = q0
    for _ in range(50):
        ik(target[0], wrist_quat(0))
    max_dq[0] = args.max_vel / fps
    q_start = dk.qpos[qa].copy()

    def reset():
        mujoco.mj_resetData(m, d)
        d.qpos[:7] = H.BASE; d.qpos[qa] = q_start
        H.set_fingers(m, d, F_OPEN)
        d.qpos[obj_q:obj_q + 3] = t[0]
        d.qpos[obj_q + 3:obj_q + 7] = Rotation.from_matrix(R[0]).as_quat()[[3, 0, 1, 2]]
        d.eq_active[weld] = 0
        glue_collisions(False)
        dk.qpos[:] = d.qpos
        mujoco.mj_forward(m, d)
        for _ in range(10):
            step(q_start, F_OPEN)

    def run(show):
        reset()
        close, closing, welded_at, released_at, last = 0.0, None, None, None, np.inf
        extra = np.zeros(len(FINGERS))                    # 挤压阶段每根手指在规划角度上多合的量
        frozen = {}                                       # 碰到物体的手指: 停在碰到时的角度, 不再往里推 (否则把物体推走)
        obj_start = None                                  # 开始合拢时物体的位置
        hold_cmd = last_hold = None                       # 焊着时的手指目标 (贴面控制)
        log = {k: [] for k in ("arm", "fing", "obj_err", "obj_z", "weld", "err", "gaps")}
        rel = None                                        # 焊上时: 抓取点和手腕朝向在物体坐标系里
        squeezed_at = None
        corr = np.zeros(3)                                # 积分修正 (位置电机带负载有几 mm 的稳态误差, 会让拇指够不到)
        for i in range(n):
            if d.eq_active[weld] and rel is not None:     # 焊着: 让物体走粘合轨迹
                g_o, q_o = rel
                qo = Rotation.from_matrix(R[i]).as_quat()[[3, 0, 1, 2]]
                qw = np.zeros(4); mujoco.mju_mulQuat(qw, qo, q_o)
                ik(R[i] @ g_o + t[i] + corr, qw)
            else:
                ik(target[i] + corr, wrist_quat(i))
            if released_at is not None and i == released_at + CLOSE + BLEND:   # 手指张开、手抬起之后再恢复碰撞
                glue_collisions(False)
            goal = R[i] @ rel[0] + t[i] if d.eq_active[weld] and rel is not None else target[i]
            if contact[i]:
                if closing is None:
                    dist = np.linalg.norm(grasp_point(d) - target[i])
                    if dist < ARRIVE or (dist < 0.06 and last - dist < STALL) or i - (s - a) >= WAIT:
                        closing = i
                    last = dist
                if closing is not None:
                    close = smoothstep((i - closing + 1) / CLOSE)
                    if obj_start is None:
                        obj_start = d.xpos[obj].copy()
                    if not d.eq_active[weld]:
                        for k in touched():
                            frozen.setdefault(k, d.qpos[fq[k]])
                if closing is not None and i - closing + 1 >= CLOSE and not d.eq_active[weld]:
                    # 挤压: 规划里贴着、实际 (物理接触) 还没碰到的手指继续合, 直到碰到
                    want = [k for k in plan_touch if k not in frozen]
                    for k in want:
                        extra[k] = min(extra[k] + SQ_STEP, F_CLOSED[k] - f_grasp[k])
                    gaps = GP.finger_gaps(m, d, groups)
                    near = all(gaps[GP.NAMES[k]] < 0.002 for k in plan_touch)
                    slid = np.linalg.norm(d.xpos[obj] - obj_start) > 0.002
                    if not want or near or slid or i - closing + 1 >= CLOSE + SQUEEZE:
                        squeezed_at = i
                if closing is not None and i - closing + 1 >= CLOSE and not d.eq_active[weld] and squeezed_at is not None:
                    set_weld(); welded_at = i; glue_collisions(True)
                    # 焊上时 实际物体 -> 实际手 的相对位姿 (和焊接约束一致); 之后手的目标 = 物体粘合轨迹 x 它
                    # IK 状态同步成实际手臂 (手可能压在物体上没到 IK 的解), 之后从这里平滑出发
                    Ro = d.xmat[obj].reshape(3, 3)
                    qinv, q_o = np.zeros(4), np.zeros(4)
                    mujoco.mju_negQuat(qinv, d.xquat[obj]); mujoco.mju_mulQuat(q_o, qinv, d.xquat[hand])
                    rel = (Ro.T @ (grasp_point(d) - d.xpos[obj]), q_o)
                    dk.qpos[qa] = d.qpos[qa]
            else:
                if d.eq_active[weld]:
                    d.eq_active[weld] = 0; released_at = i     # 碰撞先不恢复: 手指还插在物体里, 一恢复会把物体弹飞
                if closing is not None:                   # 张开
                    close = smoothstep(1 - (i - released_at + 1) / CLOSE) if released_at is not None else 0.0
            f_cmd = F_OPEN + close * (f_grasp + extra - F_OPEN)
            for k, v in frozen.items():
                f_cmd[k] = min(f_cmd[k], v) if F_CLOSED[k] >= F_OPEN[k] else max(f_cmd[k], v)
            if d.eq_active[weld]:
                # 焊着 (物体和机器人不碰撞, 相对位姿固定): 手指停在 "按实际手-物体位姿刚好贴上表面" 的角度
                if hold_cmd is None:
                    hold_cmd = contact_angles(f_cmd)
                f_cmd = hold_cmd
                last_hold = hold_cmd.copy()
            elif released_at is not None and last_hold is not None:   # 松手: 从贴面角度平滑张开
                f_cmd = F_OPEN + close * (last_hold - F_OPEN)
            step(dk.qpos[qa], f_cmd)
            corr = np.clip(corr + K_I * (goal - grasp_point(d)), -I_MAX, I_MAX)
            if welded_at is not None and (i == welded_at or (d.eq_active[weld] and (i - welded_at) % 10 == 0)):
                log["gaps"].append((i, GP.finger_gaps(m, d, groups)))
            log["arm"].append(d.qpos[qa].copy()); log["fing"].append(d.qpos[fq].copy())
            log["obj_err"].append(np.linalg.norm(d.xpos[obj] - t[i])); log["obj_z"].append(d.xpos[obj][2])
            log["weld"].append(bool(d.eq_active[weld])); log["err"].append(np.linalg.norm(grasp_point(d) - goal))
            if not show(i):
                break
        gaps = log.pop("gaps")
        L = {k: np.array(v) for k, v in log.items()}
        if gaps:
            i0, g0 = gaps[0]
            print(f"焊上那帧 ({a + i0}) 手指到物体表面 (mm, 负 = 压进, inf = 没碰): "
                  + ", ".join(f"{k} {v * 1000:.1f}" for k, v in g0.items()))
            on = [GP.NAMES[k] for k in plan_touch]
            worst = max(abs(gg[k]) for _, gg in gaps for k in on) if on else np.nan
            print(f"焊着期间 (每 10 帧查一次, {len(gaps)} 次): 规划贴着的手指 {on} 间隙 |max| {worst * 1000:.2f} mm")
        dt = 1 / fps
        va_ = np.diff(L["arm"], axis=0) / dt; vf = np.diff(L["fing"], axis=0) / dt
        th, fi = touching()
        c = contact[:len(L["weld"])]
        print(f"焊接: 第 {a + welded_at if welded_at is not None else None} 帧接上 (接触段从 {s}), "
              f"松开于第 {a + released_at if released_at is not None else None} 帧; 接触段里焊着的比例 "
              f"{L['weld'][c].mean():.0%} (从焊上到段尾 {L['weld'][welded_at - 0:e - a].mean() if welded_at is not None else 0:.0%})")
        held = L["weld"]
        print(f"物体相对粘合轨迹: 焊着时 mean {L['obj_err'][held].mean() * 100:.1f} cm, max {L['obj_err'][held].max() * 100:.1f} cm; "
              f"全程 max {L['obj_err'].max() * 100:.1f} cm; 抓取点跟踪误差 mean {L['err'].mean() * 100:.1f} cm")
        if released_at is not None:
            zr = L["obj_z"][released_at]
            print(f"松手: 物体当时高 {(zr - t[0, 2]) * 100:.1f} cm (相对起始), 结束时 {(L['obj_z'][-1] - t[0, 2]) * 100:.1f} cm")
        print(f"手臂关节: 最大速度 {np.abs(va_).max():.2f} rad/s, 最大加速度 {np.abs(np.diff(va_, axis=0) / dt).max():.1f} rad/s²; "
              f"手指: 最大速度 {np.abs(vf).max():.2f} rad/s")
        print(f"物体最高抬起 {(L['obj_z'].max() - t[0, 2]) * 100:.1f} cm (人: {(t[:, 2].max() - t[0, 2]) * 100:.1f} cm)")
        return L

    if args.video:
        import imageio, cv2
        r = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera(); cam.lookat[:] = [H.OBJ_XY[0], H.OBJ_XY[1], H.TABLE_TOP + 0.08]
        cam.distance, cam.azimuth, cam.elevation = 0.9, 150, -20
        frames = []
        src = None
        if args.src_video:
            src = cv2.VideoCapture(args.src_video); src.set(cv2.CAP_PROP_POS_FRAMES, a)

        def show(i):
            r.update_scene(d, cam); img = r.render()
            if src is not None:
                ok, raw = src.read()
                if ok:
                    cv2.putText(raw, f"{a + i}{' contact' if contact[i] else ''}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    raw = cv2.resize(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB), (raw.shape[1] * 480 // raw.shape[0], 480))
                    img = np.hstack([raw, img])
            frames.append(img); return True
        run(show)
        imageio.mimsave(args.video, frames, fps=fps, macro_block_size=1)
        print("saved", args.video)
    else:
        from mujoco import viewer as mjv
        with mjv.launch_passive(m, d) as viewer:
            while viewer.is_running():
                run(lambda i: (viewer.sync(), time.sleep(1 / fps), viewer.is_running())[2])
                time.sleep(0.5)


if __name__ == "__main__":
    main()
