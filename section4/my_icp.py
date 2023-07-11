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

# ソース点群とターゲット点群の対応付け
pcd_tree = o3d.geometry.KDTreeFlann(pcd_trg_dwn)

idx_list = []
for p in pcd_src_dwn.points:
    [_, idx, _] = pcd_tree.search_knn_vector_3d(p, 1)
    idx_list.append(idx[0])

np_pcd_trg = np.asarray(pcd_trg_dwn.points)
np_pcd_y = np_pcd_trg[idx_list].copy()


def GetCorrespondenceLines(pcd_src, pcd_trg, idx_list):
    """ 対応点の可視化関数 """
    np_pcd_src = np.asarray(pcd_src.points)
    np_pcd_trg = np.asarray(pcd_trg.points)
    np_pcd_pair = np.concatenate((np_pcd_src, np_pcd_trg))

    # 始点と終点の index のリストを生成
    num_points = len(pcd_src.points)
    lines = [[i_s, num_points + i_e] for i_s, i_e in enumerate(idx_list)]

    # LineSet を生成
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np_pcd_pair),
        lines=o3d.utility.Vector2iVector(lines)
    )

    return line_set


line_set = GetCorrespondenceLines(pcd_src_dwn, pcd_trg_dwn, idx_list)
o3d.visualization.draw_geometries([pcd_src_dwn, pcd_trg_dwn, line_set])
