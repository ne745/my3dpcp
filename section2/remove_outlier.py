import sys

import open3d as o3d


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print('Showing outliers (red) and inliers (gray): ')
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries(
        [inlier_cloud, outlier_cloud],
        zoom=0.3412,
        front=[0.4257, -0.2125, -0.8795],
        lookat=[2.6172, 2.0475, 1.532],
        up=[-0.0694, -0.9768, 0.2024]
    )


def main():
    fpth = sys.argv[1]

    print(f'Loading a point cloud from {fpth}')
    pcd = o3d.io.read_point_cloud(fpth)
    print(pcd)

    print('statistical outlier removal')
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    display_inlier_outlier(pcd, ind)

    print('Radius outlier removal')
    cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.02)
    display_inlier_outlier(pcd, ind)


if __name__ == '__main__':
    main()
