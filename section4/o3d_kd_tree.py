import copy

import numpy as np
import open3d as o3d

pcd = o3d.io.read_point_cloud('./data/Bunny.ply')
pcd.paint_uniform_color([0.5, 0.5, 0.5])

pcd_tree = o3d.geometry.KDTreeFlann(pcd)

query = 9013
pcd.colors[query] = [1, 0, 0]

# クエリの k 近傍点を抽出
[k, idx, d] = pcd_tree.search_knn_vector_3d(pcd.points[query], knn=200)
pcd_res = copy.deepcopy(pcd)
np.asarray(pcd_res.colors)[idx[1:], :] = [0, 1, 0]
o3d.visualization.draw_geometries([pcd_res])

# 指定した半径の値以内の点を抽出
[k, idx, d] = pcd_tree.search_radius_vector_3d(pcd.points[query], radius=0.01)
pcd_res = copy.deepcopy(pcd)
np.asarray(pcd_res.colors)[idx[1:], :] = [0, 1, 0]
o3d.visualization.draw_geometries([pcd_res])

# 指定した半径の値以内の点を抽出
[k, idx, d] = pcd_tree.search_hybrid_vector_3d(
    pcd.points[query], radius=0.01, max_nn=200)
pcd_res = copy.deepcopy(pcd)
np.asarray(pcd_res.colors)[idx[1:], :] = [0, 1, 0]
o3d.visualization.draw_geometries([pcd_res])
