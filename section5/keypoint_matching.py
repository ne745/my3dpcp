import pathlib

import numpy as np
import open3d as o3d


def extract_keypoint_and_feature(pcd, voxel_size):
    keypoints = pcd.voxel_down_sample(voxel_size)

    viewpoint = np.array([0, 0, 0], dtype='float64')
    radius_keypoints = 2 * voxel_size
    keypoints.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_keypoints, max_nn=30))
    keypoints.orient_normals_towards_camera_location(viewpoint)

    radius_feature = 5 * voxel_size
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        keypoints,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100))

    return keypoints, feature


def ratio_test(src_feature, trg_feature, threshold=0.9):
    np_src_feature = src_feature.data.T
    np_trg_feature = trg_feature.data.T

    corrs = o3d.utility.Vector2iVector()
    for src_idx, feature in enumerate(np_src_feature):
        distance = np.linalg.norm(np_trg_feature - feature, axis=1)
        nearest_trg_idx = np.argmin(distance)
        sorted_dist = np.sort(distance)
        ratio = sorted_dist[0] / sorted_dist[1]
        if ratio < threshold:
            corr = np.array([[src_idx], [nearest_trg_idx]], np.int32)
            corrs.append(corr)

    return corrs


def main():
    dpth_data = pathlib.Path('./data')
    pcd_src = o3d.io.read_point_cloud(str(dpth_data / 'cloud_bin_0.pcd'))
    pcd_trg = o3d.io.read_point_cloud(str(dpth_data / 'cloud_bin_1.pcd'))

    pcd_src.paint_uniform_color([0.5, 0.5, 1.0])
    pcd_trg.paint_uniform_color([1.0, 0.5, 0.5])

    init_transformation = np.identity(4)
    init_transformation[0, 3] = -3.0
    pcd_src.transform(init_transformation)

    voxel_size = 0.1
    src_kp, src_feature = extract_keypoint_and_feature(pcd_src, voxel_size)
    trg_kp, trg_feature = extract_keypoint_and_feature(pcd_trg, voxel_size)

    # src_kp.paint_uniform_color([0, 1, 0])
    # trg_kp.paint_uniform_color([0, 1, 0])
    # o3d.visualization.draw_geometries([pcd_src, src_kp, pcd_trg, trg_kp])

    corrs = ratio_test(src_feature, trg_feature)
    print('対応点の数: ', len(corrs))


if __name__ == '__main__':
    main()
