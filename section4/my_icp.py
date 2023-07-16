import copy

import numpy as np
import open3d as o3d


class ICP_Registration_Point2Point:
    def __init__(self, pcd_src, pcd_trg):
        self.pcd_src = copy.deepcopy(pcd_src)
        self.pcd_trg = copy.deepcopy(pcd_trg)
        self.np_pcd_src = np.asarray(self.pcd_src.points)
        self.np_pcd_trg = np.asarray(self.pcd_trg.points)
        self.pcd_tree = o3d.geometry.KDTreeFlann(self.pcd_trg)

        self.num_iteration = 10
        self.distance = []
        self.th_distance = 0.001
        self.th_ratio = 0.999

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

    def get_correspondence_lines(self, idx_list):
        """ 対応点の可視化関数 """
        np_pcd_pair = np.concatenate((self.np_pcd_src, self.np_pcd_trg))

        # 始点と終点の index のリストを生成
        num_points = self.np_pcd_src.shape[0]
        lines = [[i_s, num_points + i_e] for i_s, i_e in enumerate(idx_list)]

        # LineSet を生成
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np_pcd_pair),
            lines=o3d.utility.Vector2iVector(lines)
        )

        return line_set

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

        # # 対応付けの可視化
        # line_set = self.get_correspondence_lines(idx_list)
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

        return transformation

    def registration(self):
        pcd_b = copy.deepcopy(self.pcd_src)
        o3d.visualization.draw_geometries([self.pcd_src, self.pcd_trg])

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

        self.pcd_src.paint_uniform_color([1.0, 0.0, 0.0])
        o3d.visualization.draw_geometries([self.pcd_src, self.pcd_trg, pcd_b])


def main():
    pcd_src = o3d.io.read_point_cloud('./data/bun000.pcd')
    pcd_trg = o3d.io.read_point_cloud('./data/bun045.pcd')

    voxel_size = 0.005
    pcd_src_dwn = pcd_src.voxel_down_sample(voxel_size=voxel_size)
    pcd_trg_dwn = pcd_trg.voxel_down_sample(voxel_size=voxel_size)

    pcd_src_dwn.paint_uniform_color([0.0, 1.0, 0.0])
    pcd_trg_dwn.paint_uniform_color([0.0, 0.0, 1.0])

    reg = ICP_Registration_Point2Point(pcd_src_dwn, pcd_trg_dwn)
    reg.registration()


if __name__ == '__main__':
    main()
