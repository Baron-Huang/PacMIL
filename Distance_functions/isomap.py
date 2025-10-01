import torch


# 计算欧几里德距离矩阵
def euclidean_distance_matrix(X):
    n = X.shape[0]
    XX = torch.mm(X, X.t())
    X_sqnorms = torch.diag(XX)
    distances = torch.sqrt(X_sqnorms.unsqueeze(0) + X_sqnorms.unsqueeze(1) - 2 * XX)
    return distances


# 构建近邻图
def build_knn_graph(X, k):
    distances = euclidean_distance_matrix(X)
    knn_indices = torch.topk(distances, k + 1, largest=False)[1][:, 1:]  # 去掉每个点到自身的距离
    return knn_indices


# 计算最短路径距离
def shortest_path_distances(X, k):
    knn_indices = build_knn_graph(X, k)
    n = X.shape[0]
    distances = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            distances[i, j] = distances[j, i] = shortest_path_distance(X, i, j, knn_indices)
    return distances


# 单点对的最短路径距离（使用Dijkstra算法）
def shortest_path_distance(X, start, end, knn_indices):
    n = X.shape[0]
    distances = torch.full((n,), float('inf'))
    distances[start] = 0
    visited = torch.zeros(n, dtype=torch.bool)
    for _ in range(n):
        min_dist = float('inf')
        u = -1
        for i in range(n):
            if not visited[i] and distances[i] < min_dist:
                min_dist = distances[i]
                u = i
        if u == end:
            break
        visited[u] = True
        for v in knn_indices[u]:
            if not visited[v]:
                dist = torch.dist(X[u], X[v])
                distances[v] = min(distances[v], distances[u] + dist)
    return distances[end]


# ISOMAP算法
def isomap(X, n_components, k):
    distances = shortest_path_distances(X, k)
    distances = torch.nan_to_num(distances, posinf=999, neginf=999, nan=0)
    # print("distances.shape")
    # print(distances)
    H = torch.eye(X.shape[0]) - 1 / X.shape[0] * torch.ones(X.shape[0], X.shape[0])
    H = torch.nan_to_num(H, posinf=999, neginf=999, nan=0)
    # print("H.shape")
    # print(H)
    B = -0.5 * H.mm(distances ** 2).mm(H)

    # 使用SVD近似求解特征向量
    # _,_,U = torch.svd_lowrank(B,q=n_components)
    # X_iso = B @ U
    B = torch.nan_to_num(B, posinf=999, neginf=999, nan=0)
    # print("B.shape")
    # print(B)
    _, _, U = torch.linalg.svd(B)
    X_iso = B @ U[:n_components,:].T

    return X_iso


# 运行ISOMAP算法
if __name__ == '__main__':
    X = torch.rand(16,768)
    print(X.type)
    n_components = 3
    X_iso = isomap(X, n_components, 10)

    # Isomap = manifold.Isomap(n_components=3, n_neighbors=5)
    # X_r = Isomap.transform(X)
    # print(X_r)

    print(X_iso.type)


