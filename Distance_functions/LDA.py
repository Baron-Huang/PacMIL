import torch
import torch.nn.functional as F
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# 创建一个示例分类数据集
X, y = make_classification(n_samples=1000, n_features=5, n_classes=3, n_clusters_per_class=1, random_state=42)
X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.long)


# 计算类内均值向量
def compute_class_means(X, y):
    class_means = []
    for c in torch.unique(y):
        class_means.append(torch.mean(X[y == c], dim=0))
    return torch.stack(class_means)


# 计算类内散度矩阵
def compute_within_class_scatter_matrix(X, y):
    class_means = compute_class_means(X, y)
    num_classes = len(class_means)
    scatter_matrix = torch.zeros(X.shape[1], X.shape[1], dtype=torch.float)
    for c in range(num_classes):
        class_scatter = (X[y == c] - class_means[c]).t().mm(X[y == c] - class_means[c])
        scatter_matrix += class_scatter
    return scatter_matrix


# 计算类间散度矩阵
def compute_between_class_scatter_matrix(X, y):
    class_means = compute_class_means(X, y)
    overall_mean = torch.mean(X, dim=0)
    num_classes = len(class_means)
    scatter_matrix = torch.zeros(X.shape[1], X.shape[1], dtype=torch.float)
    for c in range(num_classes):
        class_mean_diff = class_means[c] - overall_mean
        class_scatter = X[y == c].shape[0] * class_mean_diff.unsqueeze(1).mm(class_mean_diff.unsqueeze(0))
        scatter_matrix += class_scatter
    return scatter_matrix


# LDA算法
def lda(X, y, n_components):
    Sw = compute_within_class_scatter_matrix(X, y)
    Sb = compute_between_class_scatter_matrix(X, y)

    # 使用梯度下降近似求解广义特征值问题
    eigenvalues, eigenvectors = torch.eig(torch.pinverse(Sw).mm(Sb), eigenvectors=True)

    # 对特征值按降序排序，选择前n_components个特征向量
    indices = torch.argsort(eigenvalues[:, 0], descending=True)[:n_components]
    W = eigenvectors[:, indices]

    return W


# 运行LDA算法并降维
n_components = 2
W = lda(X, y, n_components)
X_lda = X.mm(W)

# 可视化降维后的数据
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title('LDA')
plt.show()
