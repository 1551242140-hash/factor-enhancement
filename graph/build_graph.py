"""
图构建模块

支持：
1. 基于因子欧氏距离的 kNN 图
2. 基于因子余弦相似度的 kNN 图
3. 基于历史收益相关性的图
4. 面板数据逐期构建动态图

说明：
- 单期输入：
    X_t: (N, K)
    return_window: (L, N)
- 单期输出：
    A_t: (N, N)
- 面板输出：
    A: (T, N, N)
"""

from typing import Optional
import numpy as np

from graph.graph_utils import (
    symmetrize_adjacency,
    row_normalize,
    add_self_loops,
    normalize_adjacency,
)


def _pairwise_euclidean_distance(X: np.ndarray) -> np.ndarray:
    """
    计算两两欧氏距离矩阵

    参数
    ----
    X : ndarray, shape = (N, K)

    返回
    ----
    dist : ndarray, shape = (N, N)
    """
    x_norm = np.sum(X ** 2, axis=1, keepdims=True)
    dist_sq = x_norm + x_norm.T - 2.0 * (X @ X.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    dist = np.sqrt(dist_sq)
    return dist


def _pairwise_cosine_similarity(X: np.ndarray) -> np.ndarray:
    """
    计算两两余弦相似度矩阵

    参数
    ----
    X : ndarray, shape = (N, K)

    返回
    ----
    sim : ndarray, shape = (N, N)
    """
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm = np.where(norm < 1e-12, 1.0, norm)
    X_norm = X / norm
    sim = X_norm @ X_norm.T
    sim = np.clip(sim, -1.0, 1.0)
    return sim


def _build_knn_from_score(
    score: np.ndarray,
    k: int,
    larger_is_better: bool = True,
    self_loop: bool = False,
) -> np.ndarray:
    """
    根据打分矩阵构建 kNN 邻接矩阵

    参数
    ----
    score : ndarray, shape = (N, N)
        两两打分矩阵；若 larger_is_better=True，则取分数最大的 k 个邻居
    k : int
        邻居数量
    larger_is_better : bool
        True 表示分数越大越相似；False 表示分数越小越相似
    self_loop : bool
        是否允许节点将自己选为邻居

    返回
    ----
    A : ndarray, shape = (N, N)
        稀疏邻接矩阵（带权）
    """
    if score.ndim != 2 or score.shape[0] != score.shape[1]:
        raise ValueError("score 必须是方阵")

    n_stocks = score.shape[0]
    if k <= 0 or k >= n_stocks:
        raise ValueError(f"k 必须满足 1 <= k < N，当前 k={k}, N={n_stocks}")

    A = np.zeros((n_stocks, n_stocks), dtype=float)

    for i in range(n_stocks):
        score_i = score[i].copy()

        if not self_loop:
            if larger_is_better:
                score_i[i] = -np.inf
            else:
                score_i[i] = np.inf

        if larger_is_better:
            neighbor_idx = np.argpartition(-score_i, kth=k - 1)[:k]
        else:
            neighbor_idx = np.argpartition(score_i, kth=k - 1)[:k]

        A[i, neighbor_idx] = score_i[neighbor_idx]

    return A


def build_factor_knn_graph(
    X_t: np.ndarray,
    k: int = 10,
    tau: float = 1.0,
    symmetrize: bool = True,
    add_self_loop: bool = False,
    row_norm: bool = True,
    gcn_norm: bool = False,
) -> np.ndarray:
    """
    基于因子欧氏距离构建 kNN 图

    步骤：
    1. 计算股票之间的欧氏距离
    2. 选取距离最近的 k 个邻居
    3. 用 RBF 核函数把距离转成边权重：
        w_ij = exp(-d_ij^2 / tau^2)

    参数
    ----
    X_t : ndarray, shape = (N, K)
        单期股票因子
    k : int
        每个节点的邻居数量
    tau : float
        距离核带宽
    symmetrize : bool
        是否对称化
    add_self_loop : bool
        是否加入自环
    row_norm : bool
        是否做行归一化
    gcn_norm : bool
        是否做 GCN 对称归一化

    返回
    ----
    A_t : ndarray, shape = (N, N)
        单期邻接矩阵
    """
    if X_t.ndim != 2:
        raise ValueError(f"X_t 应为二维数组 (N, K)，当前维度为 {X_t.ndim}")
    if tau <= 0:
        raise ValueError("tau 必须大于 0")

    dist = _pairwise_euclidean_distance(X_t)

    # 基于“距离越小越好”选 kNN
    raw_knn = _build_knn_from_score(
        score=dist,
        k=k,
        larger_is_better=False,
        self_loop=False,
    )

    # 将距离转换为权重
    A_t = np.zeros_like(raw_knn)
    mask = raw_knn > 0
    A_t[mask] = np.exp(-(raw_knn[mask] ** 2) / (tau ** 2))

    if symmetrize:
        A_t = symmetrize_adjacency(A_t, method="mean")

    if add_self_loop:
        A_t = add_self_loops(A_t)

    if gcn_norm:
        A_t = normalize_adjacency(A_t)
    elif row_norm:
        A_t = row_normalize(A_t)

    return A_t


def build_factor_cosine_graph(
    X_t: np.ndarray,
    k: int = 10,
    symmetrize: bool = True,
    add_self_loop: bool = False,
    row_norm: bool = True,
    gcn_norm: bool = False,
) -> np.ndarray:
    """
    基于因子余弦相似度构建 kNN 图

    参数
    ----
    X_t : ndarray, shape = (N, K)
        单期股票因子
    k : int
        每个节点的邻居数量
    symmetrize : bool
        是否对称化
    add_self_loop : bool
        是否加入自环
    row_norm : bool
        是否行归一化
    gcn_norm : bool
        是否 GCN 对称归一化

    返回
    ----
    A_t : ndarray, shape = (N, N)
    """
    if X_t.ndim != 2:
        raise ValueError(f"X_t 应为二维数组 (N, K)，当前维度为 {X_t.ndim}")

    sim = _pairwise_cosine_similarity(X_t)

    A_t = _build_knn_from_score(
        score=sim,
        k=k,
        larger_is_better=True,
        self_loop=False,
    )

    # 余弦相似度可能为负，图权重通常不希望为负，可截断
    A_t = np.maximum(A_t, 0.0)

    if symmetrize:
        A_t = symmetrize_adjacency(A_t, method="mean")

    if add_self_loop:
        A_t = add_self_loops(A_t)

    if gcn_norm:
        A_t = normalize_adjacency(A_t)
    elif row_norm:
        A_t = row_normalize(A_t)

    return A_t


def build_return_corr_graph(
    return_window: np.ndarray,
    k: Optional[int] = 10,
    threshold: Optional[float] = None,
    symmetrize: bool = True,
    add_self_loop: bool = False,
    row_norm: bool = True,
    gcn_norm: bool = False,
    use_absolute_corr: bool = False,
) -> np.ndarray:
    """
    基于历史收益相关性构图

    参数
    ----
    return_window : ndarray, shape = (L, N)
        历史收益滚动窗口，L 为窗口长度
    k : int 或 None
        取前 k 个相关性最高的邻居；若为 None，则使用 threshold
    threshold : float 或 None
        相关性阈值；若设置，则保留 corr >= threshold 的边
    symmetrize : bool
        是否对称化
    add_self_loop : bool
        是否加入自环
    row_norm : bool
        是否行归一化
    gcn_norm : bool
        是否 GCN 对称归一化
    use_absolute_corr : bool
        是否使用绝对相关系数作为相似度

    返回
    ----
    A_t : ndarray, shape = (N, N)
    """
    if return_window.ndim != 2:
        raise ValueError(
            f"return_window 应为二维数组 (L, N)，当前维度为 {return_window.ndim}"
        )

    _, n_stocks = return_window.shape
    corr = np.corrcoef(return_window, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    if use_absolute_corr:
        score = np.abs(corr)
    else:
        score = corr.copy()

    np.fill_diagonal(score, 0.0)

    if k is not None:
        A_t = _build_knn_from_score(
            score=score,
            k=k,
            larger_is_better=True,
            self_loop=False,
        )
        A_t = np.maximum(A_t, 0.0)
    else:
        if threshold is None:
            raise ValueError("当 k 为 None 时，必须提供 threshold")
        A_t = np.where(score >= threshold, score, 0.0)
        A_t = np.maximum(A_t, 0.0)

    if symmetrize:
        A_t = symmetrize_adjacency(A_t, method="mean")

    if add_self_loop:
        A_t = add_self_loops(A_t)

    if gcn_norm:
        A_t = normalize_adjacency(A_t)
    elif row_norm:
        A_t = row_normalize(A_t)

    return A_t


def build_dynamic_graphs_from_factors(
    X: np.ndarray,
    graph_type: str = "factor_knn",
    k: int = 10,
    tau: float = 1.0,
    symmetrize: bool = True,
    add_self_loop: bool = False,
    row_norm: bool = True,
    gcn_norm: bool = False,
) -> np.ndarray:
    """
    对整个因子面板逐期构建动态图

    参数
    ----
    X : ndarray, shape = (T, N, K)
        面板因子数据
    graph_type : str
        "factor_knn" 或 "cosine"

    返回
    ----
    A : ndarray, shape = (T, N, N)
        面板邻接矩阵
    """
    if X.ndim != 3:
        raise ValueError(f"X 应为三维数组 (T, N, K)，当前维度为 {X.ndim}")

    graphs = []

    for t in range(X.shape[0]):
        X_t = X[t]

        if graph_type == "factor_knn":
            A_t = build_factor_knn_graph(
                X_t=X_t,
                k=k,
                tau=tau,
                symmetrize=symmetrize,
                add_self_loop=add_self_loop,
                row_norm=row_norm,
                gcn_norm=gcn_norm,
            )
        elif graph_type == "cosine":
            A_t = build_factor_cosine_graph(
                X_t=X_t,
                k=k,
                symmetrize=symmetrize,
                add_self_loop=add_self_loop,
                row_norm=row_norm,
                gcn_norm=gcn_norm,
            )
        else:
            raise ValueError("graph_type 仅支持 'factor_knn' 或 'cosine'")

        graphs.append(A_t)

    return np.stack(graphs, axis=0)


def build_dynamic_graphs_from_returns(
    y: np.ndarray,
    lookback: int = 12,
    k: Optional[int] = 10,
    threshold: Optional[float] = None,
    symmetrize: bool = True,
    add_self_loop: bool = False,
    row_norm: bool = True,
    gcn_norm: bool = False,
    use_absolute_corr: bool = False,
) -> np.ndarray:
    """
    基于历史收益滚动窗口逐期构建动态图

    说明：
    - 第 t 期的图使用 y[max(0, t-lookback+1):t+1] 构建
    - 因此前几个时点使用较短窗口

    参数
    ----
    y : ndarray, shape = (T, N)
        收益面板
    lookback : int
        历史窗口长度

    返回
    ----
    A : ndarray, shape = (T, N, N)
    """
    if y.ndim != 2:
        raise ValueError(f"y 应为二维数组 (T, N)，当前维度为 {y.ndim}")
    if lookback <= 0:
        raise ValueError("lookback 必须大于 0")

    t_periods = y.shape[0]
    graphs = []

    for t in range(t_periods):
        start = max(0, t - lookback + 1)
        return_window = y[start:t + 1]

        A_t = build_return_corr_graph(
            return_window=return_window,
            k=k,
            threshold=threshold,
            symmetrize=symmetrize,
            add_self_loop=add_self_loop,
            row_norm=row_norm,
            gcn_norm=gcn_norm,
            use_absolute_corr=use_absolute_corr,
        )
        graphs.append(A_t)

    return np.stack(graphs, axis=0)