import pathlib

import numpy as np
import open3d as o3d


def extract_fpfh(fpth_data):
    pcd = o3d.io.read_point_cloud(str(fpth_data))
    pcd = pcd.voxel_down_sample(0.01)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.02, max_nn=10)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.03, max_nn=100)
    )
    sum_fpfh = np.sum(np.array(fpfh.data), 1)
    return sum_fpfh / np.linalg.norm(sum_fpfh)


def main():
    class_names = ['apple', 'banana', 'camera']
    num_sample = 100

    feature_train = np.zeros((len(class_names), num_sample, 33))
    feature_test = np.zeros((len(class_names), num_sample, 33))

    # 各データの特徴量を計算
    for no_cl, cl in enumerate(class_names):
        for n in range(num_sample):
            dpth_data = pathlib.Path('./data/rgbd-dataset')
            fpth_train = dpth_data / f'{cl}/{cl}_1/{cl}_1_1_{n + 1}.pcd'
            fpth_test = dpth_data / f'{cl}/{cl}_1/{cl}_1_4_{n + 1}.pcd'

            feature_train[no_cl, n] = extract_fpfh(fpth_train)
            feature_test[no_cl, n] = extract_fpfh(fpth_test)

    for i in range(len(class_names)):
        max_sim = np.zeros((3, num_sample))
        for j in range(len(class_names)):
            sim = np.dot(feature_test[i], feature_train[j].transpose())
            max_sim[j] = np.max(sim, 1)
        correct_num = (np.argmax(max_sim, 0) == i).sum()
        acc = correct_num / num_sample * 100
        print(f'Accuracy of {class_names[i]}: {acc}%')


if __name__ == '__main__':
    main()
