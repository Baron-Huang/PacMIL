import torch
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


# 计算局部切空间矩阵
def compute_local_tangent_space(X, knn_indices):
    n, k = knn_indices.shape
    local_tangent_space = torch.zeros(n, k, X.shape[1], dtype=torch.float)
    for i in range(n):
        Xi = X[knn_indices[i]]
        Xi_centered = Xi - Xi.mean(dim=0)
        local_tangent_space[i] = torch.pinverse(Xi_centered.t())  # 使用伪逆计算切空间
    return local_tangent_space


# 计算对齐矩阵
def compute_alignment_matrix(X, knn_indices):
    local_tangent_space = compute_local_tangent_space(X, knn_indices)
    n, k, d = local_tangent_space.shape
    alignment_matrix = torch.zeros(n, d, d, dtype=torch.float)
    for i in range(n):
        Xi = X[knn_indices[i]]
        Ai = torch.mm(torch.eye(k) - torch.ones(k, k) / k, local_tangent_space[i])
        AitAi = Ai.t().mm(Ai)
        alignment_matrix[i] = torch.pinverse(AitAi)  # 使用伪逆计算对齐矩阵
    return alignment_matrix


# 降维
def ltsa(X, n_components, k):
    knn_indices = build_knn_graph(X, k)
    alignment_matrix = compute_alignment_matrix(X, knn_indices)

    # 使用SVD近似求解特征向量
    _, U = torch.symeig(alignment_matrix.view(-1, alignment_matrix.size(-1)), eigenvectors=True)
    X_iso = U[:, -n_components:]
    return X_iso


# 运行LTSA算法
k = 10
n_components = 2
X_iso = ltsa(X, n_components, k)

# 可视化降维后的数据
plt.scatter(X_iso[:, 0], X_iso[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title('LTSA')
plt.show()
