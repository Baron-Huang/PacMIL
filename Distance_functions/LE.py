import torch
import numpy as np
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt

# 创建一个示例数据集，比如瑞士卷数据
X, color = make_swiss_roll(n_samples=1000, random_state=0)
X = torch.tensor(X, dtype=torch.float)


# 构建近邻图
def build_knn_graph(X, k):
    distances = torch.cdist(X, X)  # 计算欧几里德距离矩阵
    knn_indices = torch.topk(distances, k + 1, largest=False)[1][:, 1:]  # 去掉每个点到自身的距离
    return knn_indices


# 计算权重矩阵（Unnormalized方式）
def compute_unnormalized_weights(X, knn_indices):
    n, k = knn_indices.shape
    W = torch.zeros(n, n, dtype=torch.float)
    for i in range(n):
        Xi = X[knn_indices[i]] - X[i]  # 将Xi移到原点
        distances_squared = torch.sum(Xi ** 2, dim=1)
        W[i, knn_indices[i]] = 1.0 / distances_squared
    return W


# 构建拉普拉斯矩阵
def construct_laplacian(W):
    D = torch.diag(torch.sum(W, dim=1))  # 度矩阵
    L = D - W  # 拉普拉斯矩阵
    return L


# 降维
def laplacian_eigenmaps(X, n_components, k):
    knn_indices = build_knn_graph(X, k)
    W = compute_unnormalized_weights(X, knn_indices)
    L = construct_laplacian(W)

    # 使用SVD近似求解特征向量
    _, U = torch.symeig(L, eigenvectors=True)
    X_iso = U[:, 1:n_components + 1]  # 选择最小的n_components个非零特征向量
    return X_iso


# 运行拉普拉斯特征映射算法
k = 10
n_components = 2
X_iso = laplacian_eigenmaps(X, n_components, k)

# 可视化降维后的数据
plt.scatter(X_iso[:, 0], X_iso[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title('Laplacian Eigenmaps')
plt.show()
