import sys

import numpy as np
import open3d as o3d


def main():
    fpth = sys.argv[1]

    print(f'Loading a point cloud from {fpth}')
    mesh = o3d.io.read_triangle_mesh(fpth)
    print(mesh)
    o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)

    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=10.0, max_nn=10))

    print(np.asarray(pcd.normals))
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    mesh.compute_vertex_normals()
    print(np.asarray(mesh.triangle_normals))
    o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)


if __name__ == '__main__':
    main()
