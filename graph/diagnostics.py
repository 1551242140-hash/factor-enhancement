"""
图诊断模块

包含：
1. Moran's I：检验图上的空间自相关 / 邻居聚集性
2. 图平滑性（Dirichlet energy）
3. 邻居残差相关性
4. 邻居增量解释力检验（线性回归式）

说明：
- 这些检验用于回答：
  “当前数据结构下，股票邻居信息是否可能提供额外预测能力？”
"""

from typing import Dict
import numpy as np

from graph.graph_utils import compute_laplacian, row_normalize


def morans_i(y_t: np.ndarray, A_t: np.ndarray) -> float:
    """
    计算单期 Moran's I

    数学形式：
        I = (N / sum_ij w_ij) *
            [sum_ij w_ij (y_i - y_bar)(y_j - y_bar)] /
            [sum_i (y_i - y_bar)^2]

    参数
    ----
    y_t : ndarray, shape = (N,)
        单期截面收益
    A_t : ndarray, shape = (N, N)
        单期邻接矩阵（权重矩阵）

    返回
    ----
    I : float
        Moran's I 统计量
    """
    if y_t.ndim != 1:
        raise ValueError("y_t 必须是一维数组")
    if A_t.ndim != 2 or A_t.shape[0] != A_t.shape[1]:
        raise ValueError("A_t 必须是方阵")
    if A_t.shape[0] != y_t.shape[0]:
        raise ValueError("y_t 与 A_t 的节点数不一致")

    n = y_t.shape[0]
    y_centered = y_t - np.mean(y_t)

    weight_sum = np.sum(A_t)
    denom = np.sum(y_centered ** 2)

    if weight_sum < 1e-12 or denom < 1e-12:
        return 0.0

    num = np.sum(A_t * np.outer(y_centered, y_centered))
    I = (n / weight_sum) * (num / denom)
    return float(I)


def panel_morans_i(y: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    逐期计算 Moran's I

    参数
    ----
    y : ndarray, shape = (T, N)
    A : ndarray, shape = (T, N, N)

    返回
    ----
    moran_series : ndarray, shape = (T,)
    """
    if y.ndim != 2:
        raise ValueError("y 应为二维数组 (T, N)")
    if A.ndim != 3:
        raise ValueError("A 应为三维数组 (T, N, N)")
    if y.shape[0] != A.shape[0]:
        raise ValueError("y 和 A 的时间长度不一致")

    values = [morans_i(y[t], A[t]) for t in range(y.shape[0])]
    return np.array(values, dtype=float)


def graph_dirichlet_energy(y_t: np.ndarray, A_t: np.ndarray, normalized: bool = True) -> float:
    """
    计算单期图 Dirichlet energy（图平滑性）

    数学形式：
        E(y) = y^T L y

    含义：
    - 若 E(y) 较小，表示信号在图上较平滑，相邻节点取值更接近
    - 若 E(y) 较大，表示信号在图上不平滑，图传播可能不利

    参数
    ----
    y_t : ndarray, shape = (N,)
        单期截面信号（如收益、残差）
    A_t : ndarray, shape = (N, N)
        单期邻接矩阵
    normalized : bool
        是否使用归一化拉普拉斯

    返回
    ----
    energy : float
    """
    if y_t.ndim != 1:
        raise ValueError("y_t 必须是一维数组")
    if A_t.ndim != 2 or A_t.shape[0] != A_t.shape[1]:
        raise ValueError("A_t 必须是方阵")
    if A_t.shape[0] != y_t.shape[0]:
        raise ValueError("y_t 与 A_t 的节点数不一致")

    L = compute_laplacian(A_t, normalized=normalized)
    energy = float(y_t.T @ L @ y_t)
    return energy


def panel_graph_dirichlet_energy(
    y: np.ndarray,
    A: np.ndarray,
    normalized: bool = True,
) -> np.ndarray:
    """
    逐期计算图平滑性指标

    参数
    ----
    y : ndarray, shape = (T, N)
    A : ndarray, shape = (T, N, N)

    返回
    ----
    energy_series : ndarray, shape = (T,)
    """
    if y.ndim != 2:
        raise ValueError("y 应为二维数组 (T, N)")
    if A.ndim != 3:
        raise ValueError("A 应为三维数组 (T, N, N)")
    if y.shape[0] != A.shape[0]:
        raise ValueError("y 和 A 的时间长度不一致")

    values = [graph_dirichlet_energy(y[t], A[t], normalized=normalized) for t in range(y.shape[0])]
    return np.array(values, dtype=float)


def neighbor_residual_correlation(residual_t: np.ndarray, A_t: np.ndarray) -> float:
    """
    计算单期“个体残差”与“邻居平均残差”的相关性

    步骤：
    1. 先将 A_t 行归一化
    2. 计算邻居残差：
        r_neighbor = A_row @ residual_t
    3. 计算 corr(residual_t, r_neighbor)

    含义：
    - 如果该相关性显著为正，说明“自身因子未解释完的部分”
      在图邻居上仍然有结构，图方法可能有增量价值

    参数
    ----
    residual_t : ndarray, shape = (N,)
        单期残差
    A_t : ndarray, shape = (N, N)
        单期邻接矩阵

    返回
    ----
    corr_value : float
    """
    if residual_t.ndim != 1:
        raise ValueError("residual_t 必须是一维数组")
    if A_t.ndim != 2 or A_t.shape[0] != A_t.shape[1]:
        raise ValueError("A_t 必须是方阵")
    if A_t.shape[0] != residual_t.shape[0]:
        raise ValueError("residual_t 与 A_t 的节点数不一致")

    A_row = row_normalize(A_t)
    neighbor_resid = A_row @ residual_t

    x = residual_t - np.mean(residual_t)
    y = neighbor_resid - np.mean(neighbor_resid)

    denom = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
    if denom < 1e-12:
        return 0.0

    corr_value = np.sum(x * y) / denom
    return float(corr_value)


def panel_neighbor_residual_correlation(residual: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    逐期计算邻居残差相关性

    参数
    ----
    residual : ndarray, shape = (T, N)
    A : ndarray, shape = (T, N, N)

    返回
    ----
    corr_series : ndarray, shape = (T,)
    """
    if residual.ndim != 2:
        raise ValueError("residual 应为二维数组 (T, N)")
    if A.ndim != 3:
        raise ValueError("A 应为三维数组 (T, N, N)")
    if residual.shape[0] != A.shape[0]:
        raise ValueError("residual 和 A 的时间长度不一致")

    values = [neighbor_residual_correlation(residual[t], A[t]) for t in range(residual.shape[0])]
    return np.array(values, dtype=float)


def incremental_neighbor_regression_test(
    X_t: np.ndarray,
    y_t: np.ndarray,
    A_t: np.ndarray,
    fit_intercept: bool = True,
) -> Dict[str, np.ndarray]:
    """
    邻居增量解释力检验（简化版）

    思路：
    - 基准模型： y ~ X
    - 扩展模型： y ~ [X, A X]
    - 比较两者的样本内 R^2，观察邻居特征是否有增量解释力

    说明：
    - 这是一个“无显著性检验版”的简化实现
    - 优点是依赖少、能直接用于模拟实验
    - 若你后面需要，我可以再给你补 F 检验 / t 检验版本

    参数
    ----
    X_t : ndarray, shape = (N, K)
        单期原始因子
    y_t : ndarray, shape = (N,)
        单期收益
    A_t : ndarray, shape = (N, N)
        单期邻接矩阵
    fit_intercept : bool
        是否加入截距

    返回
    ----
    result : dict
        {
            "r2_base": 基准模型R2,
            "r2_full": 扩展模型R2,
            "delta_r2": 增量R2,
            "beta_base": 基准模型系数,
            "beta_full": 扩展模型系数
        }
    """
    if X_t.ndim != 2:
        raise ValueError("X_t 应为二维数组 (N, K)")
    if y_t.ndim != 1:
        raise ValueError("y_t 应为一维数组 (N,)")
    if A_t.ndim != 2 or A_t.shape[0] != A_t.shape[1]:
        raise ValueError("A_t 应为方阵")
    if X_t.shape[0] != y_t.shape[0] or A_t.shape[0] != y_t.shape[0]:
        raise ValueError("X_t、y_t、A_t 的节点数不一致")

    def _fit_ols(X_design: np.ndarray, y: np.ndarray) -> np.ndarray:
        beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
        return beta

    def _calc_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        sst = np.sum((y_true - np.mean(y_true)) ** 2)
        sse = np.sum((y_true - y_pred) ** 2)
        if sst < 1e-12:
            return 0.0
        return float(1.0 - sse / sst)

    A_row = row_normalize(A_t)
    AX_t = A_row @ X_t

    X_base = X_t
    X_full = np.concatenate([X_t, AX_t], axis=1)

    if fit_intercept:
        X_base = np.hstack([np.ones((X_base.shape[0], 1)), X_base])
        X_full = np.hstack([np.ones((X_full.shape[0], 1)), X_full])

    beta_base = _fit_ols(X_base, y_t)
    beta_full = _fit_ols(X_full, y_t)

    y_pred_base = X_base @ beta_base
    y_pred_full = X_full @ beta_full

    r2_base = _calc_r2(y_t, y_pred_base)
    r2_full = _calc_r2(y_t, y_pred_full)

    return {
        "r2_base": np.array(r2_base),
        "r2_full": np.array(r2_full),
        "delta_r2": np.array(r2_full - r2_base),
        "beta_base": beta_base,
        "beta_full": beta_full,
    }