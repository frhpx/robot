"""
机器人抓取规划 (物体静止在桌上时, 纯运动学): 找一个手腕位姿 + 每根手指的合拢角度, 使手指真的贴在物体 (点云小球) 表面上

  被 replay_glue.py 调用; 单独跑看结果:
  python grasp_plan.py --object assets/mouse_pts

搜索: 手腕朝向 = 从上往下抓的基准朝向 绕世界 z 转 YAWS, 再绕世界 x / y 倾斜 TILTS;
      抓取点 (手张开时拇指尖和食指/中指尖的中点) = 物体中心 + 偏移 OFFS_XY x OFFS_Z
每个候选:
  1. IK (位置 + 朝向, 不限速) 够不到 (误差 > 4 mm / 3°) -> 丢掉
  2. 手张开时机器人压进物体 > 2 mm 或碰到桌子 -> 丢掉
  3. 每根手指单独从张开往握紧走, 第一次碰到物体就二分找刚好贴上的角度 (间隙 < 0.5 mm); 先碰到桌子 = 这根没贴上
  4. 所有手指放到各自的角度, 手掌 / 手腕压进物体 > 2 mm 或碰桌子 -> 丢掉
  打分: 拇指贴上 3 分 + 每根贴上的其他手指 1 分 (至少拇指 + 2 根), 同分时倾斜和偏移越小越好
"""
import argparse, json, os
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

import hoi_to_mujoco as H

YAWS = np.radians(np.arange(-90, 91, 15))
TILTS = [(0, 0), (np.radians(20), 0), (np.radians(-20), 0), (0, np.radians(20)), (0, np.radians(-20))]
OFFS_XY = [-0.02, 0.0, 0.02]
OFFS_Z = [-0.01, 0.0, 0.01, 0.02, 0.03]
PEN_OBJ = 0.002            # 手掌 / 手腕最多压进物体 2 mm
GAP = 0.0005               # 手指贴上: 二分到间隙 < 0.5 mm
# Inspire 手: 主动关节下标 -> 这根手指的 body (拇指转向 0 号固定在张开值, 不参与合拢)
GROUPS = {1: ["R_thumb_proximal", "R_thumb_intermediate", "R_thumb_distal"],
          2: ["R_index_proximal", "R_index_intermediate"], 3: ["R_middle_proximal", "R_middle_intermediate"],
          4: ["R_ring_proximal", "R_ring_intermediate"], 5: ["R_pinky_proximal", "R_pinky_intermediate"]}
NAMES = {1: "拇指", 2: "食指", 3: "中指", 4: "无名指", 5: "小指"}


class Arm:
    """右臂 IK (和 hoi_to_mujoco.py 同一个: 阻尼最小二乘 + 零空间拉向 q0 + 速度限制)"""

    def __init__(self, m):
        self.m = m
        self.qa = [m.joint(j).qposadr[0] for j in H.ARM]
        self.va = [m.joint(j).dofadr[0] for j in H.ARM]
        self.hand = m.body(H.HAND["wrist"]).id
        self.lo, self.hi = m.jnt_range[[m.joint(j).id for j in H.ARM]].T

    def grasp_point(self, d):
        return d.xpos[self.hand] + d.xmat[self.hand].reshape(3, 3) @ H.GRASP_OFFSET

    def solve(self, d, p_target, quat=None, iters=10, max_dq=np.inf):
        m, qa, va, hand = self.m, self.qa, self.va, self.hand
        q_prev = d.qpos[qa].copy()
        for _ in range(iters):
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            p = self.grasp_point(d)
            Jp, Jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, Jp, Jr, p, hand)
            err, J = p_target - p, Jp[:, va]
            if quat is not None:
                qc, dr = np.zeros(4), np.zeros(3)
                mujoco.mju_negQuat(qc, d.xquat[hand]); mujoco.mju_mulQuat(qc, quat, qc)
                mujoco.mju_quat2Vel(dr, qc, 1.0)
                err, J = np.r_[err, H.W_ROT * dr], np.r_[J, H.W_ROT * Jr[:, va]]
            Jinv = J.T @ np.linalg.inv(J @ J.T + 0.05 ** 2 * np.eye(len(err)))
            d.qpos[qa] += Jinv @ err + (np.eye(7) - Jinv @ J) @ (H.K_NULL * (H.q0 - d.qpos[qa]))
        d.qpos[qa] = np.clip(q_prev + np.clip(d.qpos[qa] - q_prev, -max_dq, max_dq), self.lo, self.hi)

    def error(self, d, p_target, quat):
        mujoco.mj_kinematics(m := self.m, d)
        qc, dr = np.zeros(4), np.zeros(3)
        mujoco.mju_negQuat(qc, d.xquat[self.hand]); mujoco.mju_mulQuat(qc, quat, qc)
        mujoco.mju_quat2Vel(dr, qc, 1.0)
        return np.linalg.norm(self.grasp_point(d) - p_target), np.linalg.norm(dr)


def quat_mul(a, b):
    q = np.zeros(4); mujoco.mju_mulQuat(q, a, b); return q


def mat2quat(R):
    return Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]


class Contacts:
    """机器人各 body 和 物体 / 桌子 的最小有符号距离 (负 = 压进去)"""

    def __init__(self, m):
        self.m = m
        self.obj = m.body("obj").id
        self.table = m.body("table").id
        self.robot = {b for b in range(m.nbody) if b not in (0, self.obj, self.table)}

    def __call__(self, d):
        mujoco.mj_forward(self.m, d)
        to_obj, to_table = {}, {}
        for c in d.contact[:d.ncon]:
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            for x, y in ((b1, b2), (b2, b1)):
                if x in self.robot and y == self.obj:
                    to_obj[x] = min(to_obj.get(x, np.inf), c.dist)
                if x in self.robot and y == self.table:
                    to_table[x] = min(to_table.get(x, np.inf), c.dist)
        return to_obj, to_table


def finger_gaps(m, d, groups, distmax=0.03):
    """每根手指 (碰撞网格) 到物体 (所有小球) 的最小距离, 负 = 压进去; distmax 以外记 inf"""
    obj = m.body("obj").id
    og = np.where(m.geom_bodyid == obj)[0]
    out = {}
    for k, bodies in groups.items():
        best = np.inf
        for g in np.where(np.isin(m.geom_bodyid, list(bodies)) & (m.geom_contype > 0))[0]:
            near = og[np.linalg.norm(d.geom_xpos[og] - d.geom_xpos[g], axis=1) < 0.08]
            for o in near:
                best = min(best, mujoco.mj_geomDistance(m, d, int(g), int(o), distmax, None))
        out[NAMES[k]] = best if best < distmax else np.inf
    return out


def plan_grasp(m, R_o, t_o, quat0, verbose=True):
    """物体位姿 (R_o, t_o, 世界系) 下找抓取. quat0 = 基准手腕朝向.
       返回 dict: g_obj (抓取点, 物体系), q_rel (手腕朝向, 物体系), f (手指目标角度), touch (贴上的手指), q_arm"""
    arm, cts = Arm(m), Contacts(m)
    obj_geoms = np.where(m.geom_bodyid == cts.obj)[0]
    margin0 = m.geom_margin[obj_geoms].copy()
    m.geom_margin[obj_geoms] = 0.01                     # 1 cm 以内也生成接触, dist 就是实际间隙 (规划完恢复)
    d = mujoco.MjData(m)
    d.qpos[:7] = H.BASE
    oq = m.joint("obj_joint").qposadr[0]
    d.qpos[oq:oq + 3], d.qpos[oq + 3:oq + 7] = t_o, mat2quat(R_o)
    fq = [m.joint(j).qposadr[0] for j in H.FINGERS]
    groups = {k: {m.body(b).id for b in bs} for k, bs in GROUPS.items()}
    fingers = set().union(*groups.values())

    def finger_touch(k, f):
        """手指 k 放到角度 f[k] (其他手指张开): 和物体的最小距离 (inf = 没碰), 是否碰桌子"""
        ff = H.F_OPEN.copy(); ff[k] = f
        H.set_fingers(m, d, ff)
        to_obj, to_table = cts(d)
        return min((to_obj.get(b, np.inf) for b in groups[k]), default=np.inf), \
            any(to_table.get(b, np.inf) < 0 for b in groups[k])

    best, n_reach = None, 0
    q_seed = H.q0.copy()
    for yaw in YAWS:
        for tx, ty in TILTS:
            qw = quat_mul(mat2quat(Rotation.from_euler("zxy", [yaw, tx, ty]).as_matrix()), quat0)
            for dz in OFFS_Z:
                for dx in OFFS_XY:
                    for dy in OFFS_XY:
                        p = t_o + [dx, dy, dz]
                        d.qpos[arm.qa] = q_seed
                        H.set_fingers(m, d, H.F_OPEN)
                        arm.solve(d, p, qw, iters=60)
                        ep, er = arm.error(d, p, qw)
                        if ep > 0.004 or er > np.radians(3):
                            continue
                        n_reach += 1
                        to_obj, to_table = cts(d)
                        if min(to_obj.values(), default=np.inf) < -PEN_OBJ or min(to_table.values(), default=np.inf) < 0:
                            continue
                        f, touch = H.F_CLOSED.copy(), []
                        for k in groups:                   # 每根手指单独合拢, 找刚好贴上的角度
                            prev = H.F_OPEN[k]
                            for x in np.linspace(H.F_OPEN[k], H.F_CLOSED[k], 26)[1:]:
                                dist, tab = finger_touch(k, x)
                                if tab:
                                    f[k] = prev; break
                                if dist < 0:
                                    lo_, hi_ = prev, x
                                    for _ in range(8):
                                        mid = (lo_ + hi_) / 2
                                        if finger_touch(k, mid)[0] < 0:
                                            hi_ = mid
                                        else:
                                            lo_ = mid
                                    f[k] = lo_; touch.append(k); break
                                prev = x
                        if 1 not in touch or len(touch) < 3:
                            continue
                        H.set_fingers(m, d, f)
                        to_obj, to_table = cts(d)
                        palm = [v for b, v in to_obj.items() if b not in fingers]
                        if min(palm, default=np.inf) < -PEN_OBJ or min(to_table.values(), default=np.inf) < 0:
                            continue
                        score = 3 * (1 in touch) + len(touch) - 1 - 0.01 * np.hypot(tx, ty) * 57 - 0.1 * np.linalg.norm([dx, dy, dz]) * 100
                        if best is None or score > best[0]:
                            best = (score, p.copy(), qw.copy(), f.copy(), touch, d.qpos[arm.qa].copy())
    m.geom_margin[obj_geoms] = margin0
    if best is None:
        raise SystemExit(f"找不到抓取 ({n_reach} 个候选够得到): 放宽 YAWS / TILTS / OFFS 或检查物体位置")
    _, p, qw, f, touch, q_arm = best
    # 最终配置下各手指到物体的实际间隙
    d.qpos[arm.qa] = q_arm
    H.set_fingers(m, d, f)
    mujoco.mj_forward(m, d)
    gaps = finger_gaps(m, d, groups)
    if verbose:
        print(f"抓取规划: {n_reach} 个候选够得到; 选中 拇指 + {len(touch) - 1} 指 "
              f"({', '.join(NAMES[k] for k in touch)}), 抓取点偏物体中心 {np.round((p - t_o) * 100, 1).tolist()} cm; "
              f"手指间隙 (mm, inf = 没碰): {', '.join(f'{k} {v * 1000:.2f}' for k, v in gaps.items())}")
    q_rel = quat_mul(mat2quat(R_o.T), qw)
    return dict(g_obj=R_o.T @ (p - t_o), q_rel=q_rel, f=f, touch=touch, q_arm=q_arm, groups=groups)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", required=True)
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.object, "meta.json")))
    H.use_hand("inspire")
    H.OBJ_POS = np.array([*H.OBJ_XY, H.TABLE_TOP + meta["size"][2] / 2])
    m = H.build_model("", args.object)
    H.GRASP_OFFSET = H.calibrate_grasp_offset(m)
    arm = Arm(m)
    d = mujoco.MjData(m); d.qpos[:7] = H.BASE; d.qpos[arm.qa] = H.q0
    for _ in range(50):
        arm.solve(d, H.OBJ_POS + [0, 0, 0.01])
    plan_grasp(m, np.eye(3), H.OBJ_POS, d.xquat[arm.hand].copy())
