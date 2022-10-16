import sys
import time

import numpy as np
import open3d as o3d


def l2_norm(a, b):
    return ((a - b) ** 2).sum(axis=1)


def farthest_point_sampling(pcd, num_samples, metrics=l2_norm):
    indices = np.zeros(num_samples, dtype=np.int32)
    points = np.asarray(pcd.points)
    distance = np.zeros((num_samples, points.shape[0]), dtype=np.float32)
    min_distance = np.ones(points.shape[0]) * np.inf

    for i in range(num_samples):
        if i == 0:
            indices[i] = 0  # np.random.randint(len(points))
        else:
            indices[i] = np.argmax(min_distance)
        farthest_point = points[indices[i]]
        distance[i, :] = metrics(farthest_point, points)
        min_distance = np.minimum(min_distance, distance[i, :])

    pcd = pcd.select_by_index(indices)
    return pcd


def my_farthest_point_sampling(pcd, num_samples, metrics=l2_norm):
    points = np.asarray(pcd.points)
    # 初期化
    indices = np.zeros(num_samples, dtype=np.int32)
    remained_points_idx = np.arange(points.shape[0])
    min_distance = np.ones_like(remained_points_idx) * np.inf

    for i in range(num_samples):
        if i == 0:
            selected = 0  # np.random.randint(len(points))
        else:
            # 各点からの最小距離のうち最大となる点をサンプリング点として選択
            selected = np.argmax(min_distance)
        # サンプリング点の保存
        indices[i] = remained_points_idx[selected]
        farthest_point = points[selected]

        remained_points_idx = np.delete(remained_points_idx, selected)
        min_distance = np.delete(min_distance, selected)
        points = np.delete(points, selected, axis=0)

        # 距離の計算
        distance = metrics(farthest_point, points)

        # 各点からの最小距離の計算
        min_distance = np.minimum(distance, min_distance)

    pcd = pcd.select_by_index(indices)
    return pcd


def is_equal_pcds(pcd1, pcd2):
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)
    for p1, p2 in zip(points1, points2):
        if not all(p1 == p2):
            return False
    return True


def main():
    fpth = sys.argv[1]
    k = int(sys.argv[2])

    print(f'Loading a point cloud from {fpth}')
    pcd = o3d.io.read_point_cloud(fpth)
    print(pcd)
    # o3d.visualization.draw_geometries(
    #     [pcd], zoom=0.3412,
    #     front=[0.4257, -0.2125, -0.8795],
    #     lookat=[2.6172, 2.0475, 1.532],
    #     up=[-0.0694, -0.9768, 0.2024]
    # )

    st = time.time()
    my_sampled_pcd = my_farthest_point_sampling(pcd, k)
    et = time.time()
    print(my_sampled_pcd)
    print(et - st)
    # o3d.visualization.draw_geometries(
    #     [sampled_pcd], zoom=0.3412,
    #     front=[0.4257, -0.2125, -0.8795],
    #     lookat=[2.6172, 2.0475, 1.532],
    #     up=[-0.0694, -0.9768, 0.2024]
    # )

    st = time.time()
    sampled_pcd = farthest_point_sampling(pcd, k)
    et = time.time()
    print(sampled_pcd)
    print(et - st)
    # o3d.visualization.draw_geometries(
    #     [sampled_pcd], zoom=0.3412,
    #     front=[0.4257, -0.2125, -0.8795],
    #     lookat=[2.6172, 2.0475, 1.532],
    #     up=[-0.0694, -0.9768, 0.2024]
    # )

    print(is_equal_pcds(sampled_pcd, my_sampled_pcd))


if __name__ == '__main__':
    main()
