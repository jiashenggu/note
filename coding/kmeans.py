import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4):
        """
        初始化K-means模型参数

        参数:
        n_clusters (int): 聚类数量（K值）
        max_iter (int): 最大迭代次数
        tol (float): 收敛阈值（中心点变化容忍度）
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None  # 聚类中心点
        self.labels = None  # 样本所属簇标签

    def _initialize_centroids(self, X):
        """随机初始化聚类中心"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        return X[indices]

    def _compute_distances(self, X):
        """计算样本到所有聚类中心的欧氏距离"""
        # 利用广播机制计算距离矩阵（样本数 × 聚类数）
        return np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

    def fit(self, X):
        """
        训练K-means模型

        参数:
        X (numpy.ndarray): 训练数据，形状为（样本数，特征数）
        """
        # 初始化中心点
        self.centroids = self._initialize_centroids(X)

        for _ in range(self.max_iter):
            # 步骤1：分配样本到最近的簇
            distances = self._compute_distances(X)
            self.labels = np.argmin(distances, axis=1)

            # 步骤2：更新聚类中心
            new_centroids = np.zeros_like(self.centroids)
            for i in range(self.n_clusters):
                # 获取当前簇的所有样本
                cluster_samples = X[self.labels == i]

                # 处理空簇：随机选择一个样本作为新中心
                if len(cluster_samples) == 0:
                    new_centroids[i] = X[np.random.choice(X.shape[0])]
                else:
                    new_centroids[i] = cluster_samples.mean(axis=0)

            # 检查收敛条件
            centroid_shift = np.linalg.norm(new_centroids - self.centroids, axis=1)
            if np.all(centroid_shift <= self.tol):
                break

            self.centroids = new_centroids

    def predict(self, X):
        """预测新样本的簇归属"""
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)


# 示例使用
if __name__ == "__main__":
    # 生成样本数据（三个高斯分布簇）
    np.random.seed(42)
    cluster_1 = np.random.normal(loc=[0, 0], scale=1, size=(100, 2))
    cluster_2 = np.random.normal(loc=[5, 5], scale=1, size=(100, 2))
    cluster_3 = np.random.normal(loc=[-5, 5], scale=1, size=(100, 2))
    X = np.vstack([cluster_1, cluster_2, cluster_3])

    # 创建并训练模型
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    labels = kmeans.labels

    # 可视化结果
    plt.figure(figsize=(10, 6))

    # 绘制数据点
    plt.scatter(
        X[:, 0], X[:, 1], c=labels, cmap="viridis", edgecolor="k", s=50, alpha=0.6
    )

    # 绘制聚类中心
    plt.scatter(
        kmeans.centroids[:, 0],
        kmeans.centroids[:, 1],
        c="red",
        marker="X",
        s=200,
        edgecolor="k",
        linewidth=2,
        label="Centroids",
    )

    plt.title("K-means Clustering Result")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()
