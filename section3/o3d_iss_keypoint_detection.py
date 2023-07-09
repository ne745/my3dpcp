import argparse
import open3d as o3d
from keypoints_to_spheres import keypoints_to_spheres


parser = argparse.ArgumentParser()
parser.add_argument('fpth_data', type=str)
args = parser.parse_args()

print("Loading a point cloud from", args.fpth_data)
pcd = o3d.io.read_point_cloud(args.fpth_data)
print(pcd)

keypoints = o3d.geometry.keypoint.compute_iss_keypoints(
    pcd,
    salient_radius=0.006,
    non_max_radius=0.005,
    gamma_21=0.5,
    gamma_32=0.5)
print(keypoints)

pcd.paint_uniform_color([0.5, 0.5, 0.5])
o3d.visualization.draw_geometries([keypoints_to_spheres(keypoints), pcd])
