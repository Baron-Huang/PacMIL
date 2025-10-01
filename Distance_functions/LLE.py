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

# 计算重构权重
def compute_reconstruction_weights(X, knn_indices):
    n, k = knn_indices.shape
    W = torch.zeros(n, n, dtype=torch.float).cuda(torch.device(0))
    for i in range(n):
        Xi = X[knn_indices[i]] - X[i]  # 将Xi移到原点
        gram_matrix = torch.mm(Xi, Xi.t()).cuda(torch.device(0))  # 计算局部协方差矩阵
        # gram_matrix += torch.eye(k) * 1e-3  # 添加微小的正则项，防止矩阵奇异
        w = torch.solve(torch.ones(k).unsqueeze(1).cuda(torch.device(0)), gram_matrix)[0]  # 求解权重
        w /= torch.sum(w)  # 归一化权重
        W[i, knn_indices[i]] = w.squeeze()
    return W

# 降维
def lle(X, n_components, k):
    knn_indices = build_knn_graph(X, k)
    W = compute_reconstruction_weights(X, knn_indices)
    M = torch.eye(X.shape[0]).cuda(torch.device(0)) - W  # 构建矩阵M = (I - W)
    eigenvalues, eigenvectors = torch.symeig(torch.mm(M.t(), M), eigenvectors=True, upper=True)
    X_iso = eigenvectors[:, 1:n_components + 1]  # 选择最小的n_components个非零特征向量
    return X_iso

# 运行LLE算法


if __name__ == '__main__':
    k = 10
    n_components = 2
    X_iso = lle(X, n_components, k)

    # 可视化降维后的数据
    plt.scatter(X_iso[:, 0], X_iso[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title('LLE')
    plt.show()
