import sys

import open3d as o3d


def main():
    fpth = sys.argv[1]
    k = int(sys.argv[2])

    print(f'Loading a point cloud from {fpth}')
    mesh = o3d.io.read_triangle_mesh(fpth)
    print(mesh)
    o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)

    sampled_pcd = mesh.sample_points_poisson_disk(number_of_points=k)
    print(sampled_pcd)
    o3d.visualization.draw_geometries([sampled_pcd])


if __name__ == '__main__':
    main()
