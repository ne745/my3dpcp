import numpy as np
import open3d as o3d

pcd_src = o3d.io.read_point_cloud('./data/bun000.pcd')
pcd_trg = o3d.io.read_point_cloud('./data/bun045.pcd')

voxel_size = 0.005
pcd_src_dwn = pcd_src.voxel_down_sample(voxel_size=voxel_size)
pcd_trg_dwn = pcd_trg.voxel_down_sample(voxel_size=voxel_size)

pcd_src_dwn.paint_uniform_color([0.0, 1.0, 0.0])
pcd_trg_dwn.paint_uniform_color([0.0, 0.0, 1.0])
o3d.visualization.draw_geometries([pcd_src_dwn, pcd_trg_dwn])


def quaternion2rotation(q):
    rot = np.array([
        [q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2,
         2 * (q[1] * q[2] - q[0] * q[3]),
         2 * (q[1] * q[3] + q[0] * q[2])],

        [2 * (q[1] * q[2] + q[0] * q[3]),
         q[0] ** 2 - q[1] ** 2 + q[2] ** 2 - q[3] ** 2,
         2 * (q[2] * q[3] - q[0] * q[1])],

        [2 * (q[1] * q[3] - q[0] * q[2]),
         2 * (q[2] * q[3] + q[0] * q[1]),
         q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2],
    ])
    return rot


q = np.array([1, 0, 0, 0, 0, 0, 0], dtype=float)
rot = quaternion2rotation(q)
print(rot)
