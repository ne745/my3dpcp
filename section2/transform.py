import copy

import numpy as np
import open3d as o3d


# 軸のメッシュ
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

# 回転
R = o3d.geometry.get_rotation_matrix_from_yxz([np.pi / 3, 0, 0])
print('R:')
print(np.round(R, 7))

R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, np.pi / 3, 0])
print('R:')
print(np.round(R, 7))

R = o3d.geometry.get_rotation_matrix_from_quaternion(
    [np.cos(np.pi / 6), 0, np.sin(np.pi / 6), 0])
print('R:')
print(np.round(R, 7))

mesh_rotate = copy.deepcopy(mesh)
mesh_rotate.rotate(R, center=[0, 0, 0])
# o3d.visualization.draw_geometries([mesh, mesh_rotate])

# 並進
t = [0.5, 0.7, 0.1]
mesh_translate = copy.deepcopy(mesh_rotate)
mesh_translate.translate(t)
# o3d.visualization.draw_geometries([mesh, mesh_translate])

# 回転と並進
T = np.eye(4)
T[:3, :3] = R
T[:3, 3] = t
mesh_transformation = copy.deepcopy(mesh)
mesh_transformation.transform(T)
# o3d.visualization.draw_geometries([mesh, mesh_transformation])

# スケール
mesh_scale = copy.deepcopy(mesh)
mesh_scale.scale(0.5, center=mesh_scale.get_center())
o3d.visualization.draw_geometries([mesh, mesh_scale])
