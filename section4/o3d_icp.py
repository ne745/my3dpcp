import copy

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

threshold = 0.05
trans_init = np.identity(4)
obj_func = o3d.pipelines.registration.TransformationEstimationPointToPoint()
result = o3d.pipelines.registration.registration_icp(
    pcd_src_dwn, pcd_trg_dwn, threshold, trans_init, obj_func
)
print(result.transformation)

pcd_reg = copy.deepcopy(pcd_src_dwn).transform(result.transformation)
pcd_reg.paint_uniform_color([1.0, 0.0, 0.0])
o3d.visualization.draw_geometries([pcd_reg, pcd_trg_dwn])
