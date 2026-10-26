"""
合成测试 (不需要 GPU): 从 mesh 资产采样 "相机看得到" 的表面点, 走 object_points.py 步骤 4 之后的流程, 生成点云资产

  python make_test_pts.py                                    # assets/mouse/mesh.obj -> assets/test_pts
  python make_test_pts.py --mesh assets/mouse --name test_pts --cam-pitch 50

相机方向 = cam_to_robot_R 第 3 列 (相机朝前的 z 轴) 反向; 只保留法向和它夹角 < 90° 的点 (不考虑自遮挡).
桌面高度 = mesh 最低点.
"""
import argparse, json, os
import numpy as np
import trimesh
from object_points import HERE, build_cloud, save_asset
from hoi_to_mujoco import cam_to_robot_R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default=os.path.join(HERE, "assets", "mouse"), help="object_asset.py 的输出目录")
    ap.add_argument("--name", default="test_pts")
    ap.add_argument("--cam-pitch", type=float, default=50, help="和之后 hoi_to_mujoco.py 的 --cam-pitch 一致")
    ap.add_argument("--n-sample", type=int, default=20000)
    ap.add_argument("--voxel", type=float, default=0.007)
    ap.add_argument("--mass", type=float, default=0.1)
    args = ap.parse_args()

    mesh = trimesh.load(os.path.join(args.mesh, "mesh.obj"), force="mesh")
    meta = json.load(open(os.path.join(args.mesh, "meta.json")))
    p, face = trimesh.sample.sample_surface(mesh, args.n_sample, seed=0)
    to_cam = -cam_to_robot_R(np.deg2rad(args.cam_pitch))[:, 2]
    seen = mesh.face_normals[face] @ to_cam > 0
    p = p[seen]
    c = np.tile(meta["rgba"][:3], (len(p), 1))
    print(f"采样 {args.n_sample} 个表面点, 相机看得到 {len(p)} 个")

    q, qc, r, size, voxel, yaw = build_cloud(p, c, float(mesh.bounds[0, 2]), args.voxel)
    out = os.path.join(HERE, "assets", args.name)
    m = save_asset(out, q, qc, r, size, args.mass, meta.get("frame", -1), args.cam_pitch, voxel, yaw)
    print(f"saved {out}: {m['n']} 个点, 体素 {voxel * 1000:.1f} mm, 尺寸 {np.round(size * 100, 1).tolist()} cm "
          f"(mesh {np.round(np.array(meta['size']) * 100, 1).tolist()} cm)")


if __name__ == "__main__":
    main()
