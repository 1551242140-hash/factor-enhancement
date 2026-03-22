"""
图增强特征构造模块

目标：
1. 根据图邻接矩阵 A 和原始因子 X，构造邻居聚合特征
2. 支持一阶传播 Ax、二阶传播 A^2x
3. 支持拼接 [X, Ax] 或 [X, Ax, A^2x]
"""

from typing import List
import numpy as np


def _check_x_a_shapes(X_t: np.ndarray, A_t: np.ndarray) -> None:
    """
    检查单期特征矩阵 X_t 与邻接矩阵 A_t 的维度是否合法

    参数
    ----
    X_t : ndarray, shape = (N, K)
        单期股票因子矩阵
    A_t : ndarray, shape = (N, N)
        单期图邻接矩阵
    """
    if X_t.ndim != 2:
        raise ValueError(f"X_t 应为二维数组 (N, K)，当前维度为 {X_t.ndim}")
    if A_t.ndim != 2:
        raise ValueError(f"A_t 应为二维数组 (N, N)，当前维度为 {A_t.ndim}")

    n_stocks = X_t.shape[0]
    if A_t.shape != (n_stocks, n_stocks):
        raise ValueError(
            f"A_t 形状应为 ({n_stocks}, {n_stocks})，当前为 {A_t.shape}"
        )


def neighbor_average_features(X_t: np.ndarray, A_t: np.ndarray) -> np.ndarray:
    """
    构造一阶邻居平均特征 Ax

    数学形式：
        X_neighbor = A_t @ X_t

    若 A_t 已做行归一化，则可理解为“邻居特征加权平均”

    参数
    ----
    X_t : ndarray, shape = (N, K)
        单期原始因子
    A_t : ndarray, shape = (N, N)
        单期邻接矩阵

    返回
    ----
    X_neighbor : ndarray, shape = (N, K)
        邻居聚合后的因子特征
    """
    _check_x_a_shapes(X_t, A_t)
    return A_t @ X_t


def graph_propagation_features(
    X_t: np.ndarray,
    A_t: np.ndarray,
    num_hops: int = 2,
) -> List[np.ndarray]:
    """
    构造多阶图传播特征：[Ax, A^2x, ..., A^num_hops x]

    参数
    ----
    X_t : ndarray, shape = (N, K)
        单期原始因子
    A_t : ndarray, shape = (N, N)
        单期邻接矩阵
    num_hops : int
        图传播阶数，至少为 1

    返回
    ----
    propagated_list : list of ndarray
        长度为 num_hops 的列表
        第 h 个元素表示 A^(h+1) X_t（从 Ax 开始）
    """
    _check_x_a_shapes(X_t, A_t)

    if num_hops < 1:
        raise ValueError("num_hops 必须大于等于 1")

    propagated_list = []
    current = A_t @ X_t  # 一阶传播 Ax
    propagated_list.append(current)

    for _ in range(1, num_hops):
        current = A_t @ current
        propagated_list.append(current)

    return propagated_list


def concat_self_and_graph_features(
    X_t: np.ndarray,
    A_t: np.ndarray,
    num_hops: int = 1,
    include_self: bool = True,
) -> np.ndarray:
    """
    拼接原始特征与图传播特征

    示例：
    - num_hops = 1, include_self = True  -> [X, Ax]
    - num_hops = 2, include_self = True  -> [X, Ax, A^2x]
    - num_hops = 1, include_self = False -> [Ax]

    参数
    ----
    X_t : ndarray, shape = (N, K)
        单期原始因子
    A_t : ndarray, shape = (N, N)
        单期邻接矩阵
    num_hops : int
        图传播阶数
    include_self : bool
        是否包含原始特征 X_t

    返回
    ----
    X_concat : ndarray, shape = (N, K_new)
        拼接后的增强特征
    """
    _check_x_a_shapes(X_t, A_t)

    feature_blocks = []

    if include_self:
        feature_blocks.append(X_t)

    propagated_list = graph_propagation_features(X_t, A_t, num_hops=num_hops)
    feature_blocks.extend(propagated_list)

    X_concat = np.concatenate(feature_blocks, axis=1)
    return X_concat


def build_panel_graph_features(
    X: np.ndarray,
    A: np.ndarray,
    num_hops: int = 1,
    include_self: bool = True,
) -> np.ndarray:
    """
    对整个面板数据逐期构造图增强特征

    参数
    ----
    X : ndarray, shape = (T, N, K)
        面板因子数据
    A : ndarray, shape = (T, N, N)
        面板邻接矩阵
    num_hops : int
        图传播阶数
    include_self : bool
        是否包含原始因子

    返回
    ----
    X_graph : ndarray, shape = (T, N, K_new)
        图增强后的面板特征
    """
    if X.ndim != 3:
        raise ValueError(f"X 应为三维数组 (T, N, K)，当前维度为 {X.ndim}")
    if A.ndim != 3:
        raise ValueError(f"A 应为三维数组 (T, N, N)，当前维度为 {A.ndim}")

    t_periods, n_stocks, _ = X.shape
    if A.shape[0] != t_periods:
        raise ValueError("X 和 A 的时间长度 T 必须一致")
    if A.shape[1] != n_stocks or A.shape[2] != n_stocks:
        raise ValueError("A 的股票维度必须与 X 的股票数量一致")

    feature_list = []
    for t in range(t_periods):
        X_t_graph = concat_self_and_graph_features(
            X_t=X[t],
            A_t=A[t],
            num_hops=num_hops,
            include_self=include_self,
        )
        feature_list.append(X_t_graph)

    X_graph = np.stack(feature_list, axis=0)
    return X_graph


def build_neighbor_only_panel_features(
    X: np.ndarray,
    A: np.ndarray,
) -> np.ndarray:
    """
    构造仅包含一阶邻居聚合特征 Ax 的面板版本

    参数
    ----
    X : ndarray, shape = (T, N, K)
        面板因子数据
    A : ndarray, shape = (T, N, N)
        面板邻接矩阵

    返回
    ----
    X_neighbor : ndarray, shape = (T, N, K)
        面板邻居聚合特征
    """
    if X.ndim != 3:
        raise ValueError(f"X 应为三维数组 (T, N, K)，当前维度为 {X.ndim}")
    if A.ndim != 3:
        raise ValueError(f"A 应为三维数组 (T, N, N)，当前维度为 {A.ndim}")

    t_periods, n_stocks, _ = X.shape
    if A.shape[0] != t_periods:
        raise ValueError("X 和 A 的时间长度 T 必须一致")
    if A.shape[1] != n_stocks or A.shape[2] != n_stocks:
        raise ValueError("A 的股票维度必须与 X 的股票数量一致")

    out = np.zeros_like(X, dtype=float)
    for t in range(t_periods):
        out[t] = neighbor_average_features(X[t], A[t])

    return out