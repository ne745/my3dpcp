import sys

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
            indices[i] = np.random.randint(len(points))
        else:
            indices[i] = np.argmax(min_distance)
        farthest_point = points[indices[i]]
        distance[i, :] = metrics(farthest_point, points)
        min_distance = np.minimum(min_distance, distance[i, :])

    pcd = pcd.select_by_index(indices)
    return pcd


def main():
    fpth = sys.argv[1]
    k = int(sys.argv[2])

    print(f'Loading a point cloud from {fpth}')
    pcd = o3d.io.read_point_cloud(fpth)
    print(pcd)

    o3d.visualization.draw_geometries(
        [pcd], zoom=0.3412,
        front=[0.4257, -0.2125, -0.8795],
        lookat=[2.6172, 2.0475, 1.532],
        up=[-0.0694, -0.9768, 0.2024]
    )

    sampled_pcd = farthest_point_sampling(pcd, k)
    print(sampled_pcd)
    o3d.visualization.draw_geometries(
        [sampled_pcd], zoom=0.3412,
        front=[0.4257, -0.2125, -0.8795],
        lookat=[2.6172, 2.0475, 1.532],
        up=[-0.0694, -0.9768, 0.2024]
    )


if __name__ == '__main__':
    main()
