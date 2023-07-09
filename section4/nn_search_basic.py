import numpy as np
import open3d as o3d

# sin 関数に従う点列の生成
x = np.arange(-np.pi, np.pi, 0.1)
y = np.sin(x)
z = np.zeros_like(x)
np_sin = np.vstack([x, y, z]).T

np_p = np.array([1.0, 0.0, 0.0])

pcd_sin = o3d.geometry.PointCloud()
pcd_sin.points = o3d.utility.Vector3dVector(np_sin)
pcd_sin.paint_uniform_color([0.5, 0.5, 0.5])

pcd_p = o3d.geometry.PointCloud()
pcd_p.points = o3d.utility.Vector3dVector([np_p])
pcd_p.paint_uniform_color([0.0, 0.0, 1.0])

mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

o3d.visualization.draw_geometries([mesh, pcd_sin, pcd_p])


def calc_dist(p, X):
    dists = np.linalg.norm(p - X, axis=1)
    min_dist = min(dists)
    min_idx = np.argmin(dists)
    return min_dist, min_idx


min_dist, min_idx = calc_dist(np_p, np_sin)
np.asarray(pcd_sin.colors)[min_idx] = [0.0, 1.0, 0.0]
print(f'distance: {min_dist:.2f}, idx: {min_idx}')
print('nearest neighbor:', np_sin[min_idx])
o3d.visualization.draw_geometries([mesh, pcd_sin, pcd_p])
