import sys

import open3d as o3d


fpth = sys.argv[1]
sampling_width = 0.03

print(f'Loading a point cloud from {fpth}')
pcd = o3d.io.read_point_cloud(fpth)
print(pcd)

o3d.visualization.draw_geometries(
    [pcd],
    zoom=0.3412,
    front=[0.4257, -0.2125, -0.8795],
    lookat=[2.6172, 2.0475, 1.532],
    up=[-0.0694, -0.9768, 0.2024]
)

sampled_pcd = pcd.voxel_down_sample(voxel_size=sampling_width)
print(sampled_pcd)

o3d.visualization.draw_geometries(
    [sampled_pcd],
    zoom=0.3412,
    front=[0.4257, -0.2125, -0.8795],
    lookat=[2.6172, 2.0475, 1.532],
    up=[-0.0694, -0.9768, 0.2024]
)

voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
    sampled_pcd, voxel_size=sampling_width)
o3d.visualization.draw_geometries(
    [voxel_grid],
    zoom=0.3412,
    front=[0.4257, -0.2125, -0.8795],
    lookat=[2.6172, 2.0475, 1.532],
    up=[-0.0694, -0.9768, 0.2024]
)
