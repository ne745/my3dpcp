import sys

import numpy as np
import open3d as o3d

fpath = sys.argv[1]

print('Loading a point cloud from', fpath)
pcd = o3d.io.read_point_cloud(fpath)
# pcd = o3d.io.read_triangle_mesh(fpath)

print(pcd)
if isinstance(pcd, o3d.cpu.pybind.geometry.PointCloud):
    print(np.asarray(pcd.points))

o3d.visualization.draw_geometries([pcd])
