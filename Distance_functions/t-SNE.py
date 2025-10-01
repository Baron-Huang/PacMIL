import torch
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt

# 创建一个示例数据集，比如瑞士卷数据
X, color = make_swiss_roll(n_samples=50, random_state=0)
X = torch.tensor(X, dtype=torch.float)

# 计算高维空间中数据点之间的相似性概率分布
def compute_high_dimensional_probabilities(X, perplexity=30, epsilon=1e-5):
    n, _ = X.shape
    P = torch.zeros(n, n, dtype=torch.float)
    for i in range(n):
        distances_squared = torch.sum((X - X[i])**2, dim=1)
        exp_values = torch.exp(-distances_squared / (2 * perplexity**2))
        exp_values[i] = 0
        sum_exp = torch.sum(exp_values) + epsilon
        P[i] = exp_values / sum_exp
    return P

# 计算低维空间中数据点之间的相似性概率分布
def compute_low_dimensional_probabilities(Y):
    n, _ = Y.shape
    Q = torch.zeros(n, n, dtype=torch.float)
    for i in range(n):
        distances_squared = torch.sum((Y - Y[i])**2, dim=1)
        inv_distances_squared = 1 / (1 + distances_squared)
        inv_distances_squared[i] = 0
        sum_inv = torch.sum(inv_distances_squared)
        Q[i] = inv_distances_squared / sum_inv
    return Q

# t-SNE算法
def tsne(X, n_components=2, perplexity=30, learning_rate=100, n_iters=1000):
    n, _ = X.shape
    Y = torch.randn(n, n_components, dtype=torch.float, requires_grad=True)

    for i in range(n_iters):
        P = compute_high_dimensional_probabilities(X, perplexity)
        Q = compute_low_dimensional_probabilities(Y)

        loss = torch.sum(P * torch.log(P / Q))
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss.item()}")

        loss.backward()
        with torch.no_grad():
            Y -= learning_rate * Y.grad
            Y.grad.zero_()

    return Y

# 运行t-SNE算法
n_components = 2
perplexity = 30
learning_rate = 100
n_iters = 1000
X_iso = tsne(X, n_components, perplexity, learning_rate, n_iters).detach().cpu().numpy()

# 可视化降维后的数据
plt.scatter(X_iso[:, 0], X_iso[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title('t-SNE')
plt.show()
