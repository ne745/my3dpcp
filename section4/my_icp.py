import copy
import time

import numpy as np
import open3d as o3d


class ICP_Registraion:
    def __init__(self, pcd_src, pcd_trg):
        self.pcd_before = copy.deepcopy(pcd_src)
        self.pcd_src = copy.deepcopy(pcd_src)
        self.pcd_trg = copy.deepcopy(pcd_trg)
        self.np_pcd_src = np.asarray(self.pcd_src.points)
        self.np_pcd_trg = np.asarray(self.pcd_trg.points)
        self.pcd_tree = o3d.geometry.KDTreeFlann(self.pcd_trg)

        self.num_iteration = 10
        self.distance = []
        self.th_distance = 0.001
        self.th_ratio = 0.999

        # 過程表示用
        self.transformations = []
        self.closest_indices = []

    def get_correspondence_lines(self, pcd_s, pcd_t, idx_list):
        """ 対応点の可視化関数 """
        np_pcd_s = np.asarray(pcd_s.points)
        np_pcd_t = np.asarray(pcd_t.points)
        np_pcd_pair = np.concatenate((np_pcd_s, np_pcd_t))

        # 始点と終点の index のリストを生成
        num_points = self.np_pcd_src.shape[0]
        lines = [[i_s, num_points + i_e] for i_s, i_e in enumerate(idx_list)]

        # LineSet を生成
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np_pcd_pair),
            lines=o3d.utility.Vector2iVector(lines)
        )

        return line_set

    def visualize_icp_progress(self):
        pcd_res = copy.deepcopy(self.pcd_before)
        pcd_res.paint_uniform_color([1.0, 0.0, 0.0])
        lineset_res = o3d.geometry.LineSet()
        gen = zip(self.closest_indices, self.transformations)

        def animation(vis):
            try:
                indices, tf = next(gen)
            except StopIteration:
                return False

            pcd_res.transform(tf)
            lineset = self.get_correspondence_lines(
                pcd_res, self.pcd_trg, indices)

            lineset_res.lines = lineset.lines
            lineset_res.points = lineset.points

            vis.update_geometry(pcd_res)
            vis.update_geometry(lineset_res)

            time.sleep(0.2)

        o3d.visualization.draw_geometries_with_animation_callback(
            [self.pcd_trg, pcd_res, lineset_res], animation,
            width=640, height=500)


class ICP_Registration_Point2Point(ICP_Registraion):
    def quaternion2rotation(self, q):
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

    def find_closest_points(self):
        """ 各点の一番近い点を探す """
        # ソース点群とターゲット点群の対応付け
        idx_list = []
        distance = []
        for p in self.pcd_src.points:
            [_, idx, d] = self.pcd_tree.search_knn_vector_3d(p, 1)
            idx_list.append(idx[0])
            distance.append(d[0])
        np_pcd_y = self.np_pcd_trg[idx_list].copy()
        # 距離を終了条件に使用するため計算
        self.distance.append(np.sqrt(np.mean(np.array(distance))))

        self.closest_indices.append(idx_list)

        # # 対応付けの可視化
        # line_set = self.get_correspondence_lines(
        #     self.pcd_src, self.pcd_trg, idx_list)
        # o3d.visualization.draw_geometries(
        #     [self.pcd_src, self.pcd_trg, line_set])

        return np_pcd_y

    def compute_registration_param(self, np_pcd_y):
        # 剛体変換の推定
        mu_src = self.np_pcd_src.mean(axis=0)
        mu_y = np_pcd_y.mean(axis=0)

        covar = np.zeros((3, 3))
        for p_s, p_y in zip(self.np_pcd_src, np_pcd_y):
            covar += np.dot(p_s.reshape(-1, 1), p_y.reshape(1, -1))
        covar /= self.np_pcd_src.shape[0]
        covar -= np.dot(mu_src.reshape(-1, 1), mu_y.reshape(1, -1))

        A = covar - covar.T
        delta = np.array([A[1, 2], A[2, 0], A[0, 1]])
        tr_covar = np.trace(covar)

        Q = np.zeros((4, 4))
        Q[0, 0] = tr_covar
        Q[0, 1:4] = delta
        Q[1:4, 0] = delta
        Q[1:4, 1:4] = covar + covar.T - tr_covar * np.identity(3)

        eigen_val, eigen_vec = np.linalg.eig(Q)
        rot = self.quaternion2rotation(eigen_vec[:, np.argmax(eigen_val)])
        trans = mu_y - np.dot(rot, mu_src)

        # 同次変換行列
        transformation = np.identity(4)
        transformation[0:3, 0:3] = rot.copy()
        transformation[0:3, 3] = trans.copy()

        self.transformations.append(transformation)

        return transformation

    def registration(self):
        # o3d.visualization.draw_geometries([self.pcd_src, self.pcd_trg])

        for it in range(self.num_iteration):
            # ソース点群とターゲット点群の対応付
            np_pcd_y = self.find_closest_points()
            # 剛体変形の推定
            transformation = self.compute_registration_param(np_pcd_y)
            # 点群の更新
            self.pcd_src.transform(transformation)

            # 収束判定
            if it > 2:
                if self.distance[-1] < self.th_distance:
                    break
                if self.distance[-1] / self.distance[-2] > self.th_ratio:
                    break

        # self.pcd_src.paint_uniform_color([1.0, 0.0, 0.0])
        # o3d.visualization.draw_geometries(
        #     [self.pcd_src, self.pcd_trg, self.pcd_before])


class ICP_Registration_Point2Plane(ICP_Registraion):
    def __init__(self, pcd_src, pcd_trg):
        super().__init__(pcd_src, pcd_trg)
        self.np_normal_trg = np.asarray(self.pcd_trg.normals)

    def find_closest_points(self):
        """ 各点の一番近い点を探す """
        # ソース点群とターゲット点群の対応付け
        idx_list = []
        distance = []
        for p in self.pcd_src.points:
            [_, idx, d] = self.pcd_tree.search_knn_vector_3d(p, 1)
            idx_list.append(idx[0])
            distance.append(d[0])
        np_pcd_y = self.np_pcd_trg[idx_list].copy()
        np_normal_y = self.np_normal_trg[idx_list].copy()
        # 距離を終了条件に使用するため計算
        self.distance.append(np.sqrt(np.mean(np.array(distance))))

        self.closest_indices.append(idx_list)

        # # 対応付けの可視化
        # line_set = self.get_correspondence_lines(
        #     self.pcd_src, self.pcd_trg, idx_list)
        # o3d.visualization.draw_geometries(
        #     [self.pcd_src, self.pcd_trg, line_set])

        return np_pcd_y, np_normal_y

    def compute_rotaion_matrix(self, rotation_vector):
        """ ロドリゲスの回転公式 """
        theta = np.linalg.norm(rotation_vector)
        axis = (rotation_vector / theta).reshape(-1)
        # 歪対称行列
        w = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0]
        ])
        return np.identity(3) + np.sin(theta) * w\
            + ((1 - np.cos(theta)) * np.dot(w, w))

    def compute_transformation_matrix(self, six_deg_vector):
        """ 6 次元ベクトルを同時変換行列に変換 """
        # 6 次元ベクトルの
        # 前半 3 つ: 回転ベクトル
        # 後半 3 つ: 並進ベクトル
        rotation_vector, translation_vector = np.split(
            six_deg_vector.squeeze(), 2)

        # ロドリゲスの回転公式をより、回転ベクトルから回転行列を計算
        rotation_matrix = self.compute_rotaion_matrix(rotation_vector)

        # 同次変換行列
        transformation = np.identity(4)
        transformation[0:3, 0:3] = rotation_matrix
        transformation[0:3, 3] = translation_vector
        return transformation

    def compute_vector_step_by_step(self, np_pcd_src, np_pcd_y, np_normal_y):
        # 剛体変換の推定
        A = np.zeros((6, 6))
        b = np.zeros((6, 1))
        for i in range(len(np_pcd_src)):
            xn = np.cross(np_pcd_src[i], np_normal_y[i])
            xn_n = np.hstack((xn, np_normal_y[i])).reshape(-1, 1)
            A += np.dot(xn_n, xn_n.T)

            nT = np_normal_y[i].reshape(1, -1)
            p_x = (np_pcd_y[i] - np_pcd_src[i]).reshape(-1, 1)
            b += xn_n * np.dot(nT, p_x)
        u_opt = np.dot(np.linalg.inv(A), b)
        return u_opt

    def compute_vector(self, np_pcd_src, np_pcd_y, np_normal_y):
        # ヤコビアンと残差を計算
        num_points = len(np_pcd_y)
        J = np.zeros((num_points, 6))
        r = np.zeros(num_points)
        for i in range(num_points):
            cross_product = np.cross(np_pcd_src[i], np_normal_y[i])
            J[i, :] = np.hstack((cross_product, np_normal_y[i]))
            r[i] = np.dot(np_normal_y[i], np_pcd_y[i] - np_pcd_src[i])

        # 最小二乗法を用いて解を求める
        # 通常の方法で解くと、数値的な不安定性があるため、
        # numpy.linalg.lstsqを用いて最小二乗問題を解く。
        JtJ = np.dot(J.T, J)
        Jtr = np.dot(J.T, r)
        x, _, _, _ = np.linalg.lstsq(JtJ, Jtr, rcond=None)
        return x

    def compute_registration_param(self, np_pcd_y, np_normal_y):
        x = self.compute_vector(self.np_pcd_src, np_pcd_y, np_normal_y)
        transformation = self.compute_transformation_matrix(x)
        self.transformations.append(transformation)
        return transformation

    def registration(self):
        # o3d.visualization.draw_geometries([self.pcd_src, self.pcd_trg])

        for it in range(self.num_iteration):
            # ソース点群とターゲット点群の対応付
            np_pcd_y, np_normal_y = self.find_closest_points()
            # 剛体変形の推定
            transformation = self.compute_registration_param(
                np_pcd_y, np_normal_y)
            # 点群の更新
            self.pcd_src.transform(transformation)

            # 収束判定
            if it > 2:
                if self.distance[-1] < self.th_distance:
                    break
                if self.distance[-1] / self.distance[-2] > self.th_ratio:
                    break

        self.pcd_src.paint_uniform_color([1.0, 0.0, 0.0])
        o3d.visualization.draw_geometries(
            [self.pcd_src, self.pcd_trg, self.pcd_before])


def main():
    pcd_src = o3d.io.read_point_cloud('./data/bun000.pcd')
    pcd_trg = o3d.io.read_point_cloud('./data/bun045.pcd')

    voxel_size = 0.005
    pcd_src_dwn = pcd_src.voxel_down_sample(voxel_size=voxel_size)
    pcd_trg_dwn = pcd_trg.voxel_down_sample(voxel_size=voxel_size)

    pcd_src_dwn.paint_uniform_color([0.0, 1.0, 0.0])
    pcd_trg_dwn.paint_uniform_color([0.0, 0.0, 1.0])

    reg = ICP_Registration_Point2Plane(pcd_src_dwn, pcd_trg_dwn)
    reg.registration()

    # reg.visualize_icp_progress()


if __name__ == '__main__':
    main()
