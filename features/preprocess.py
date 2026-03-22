"""
特征预处理模块

支持：
1. 截面标准化（每个时点对股票截面做 z-score）
2. 全样本标准化（对整个面板统一做 z-score）
3. 分位数缩尾（winsorize）
4. 缺失值填充
"""

import numpy as np


def _safe_std(x: np.ndarray, axis=None, keepdims=False) -> np.ndarray:
    """
    安全计算标准差，避免出现 0 标准差导致除零问题
    """
    std = np.std(x, axis=axis, keepdims=keepdims)
    std = np.where(std < 1e-12, 1.0, std)
    return std


def zscore_cross_section(X_t: np.ndarray) -> np.ndarray:
    """
    对单期因子做横截面标准化（沿股票维度）

    参数
    ----
    X_t : ndarray, shape = (N, K)
        单期因子矩阵

    返回
    ----
    X_t_z : ndarray, shape = (N, K)
        标准化后的单期因子
    """
    if X_t.ndim != 2:
        raise ValueError(f"X_t 应为二维数组 (N, K)，当前维度为 {X_t.ndim}")

    mean = np.mean(X_t, axis=0, keepdims=True)
    std = _safe_std(X_t, axis=0, keepdims=True)
    X_t_z = (X_t - mean) / std
    return X_t_z


def zscore_panel(X: np.ndarray) -> np.ndarray:
    """
    对整个面板逐期做横截面标准化

    参数
    ----
    X : ndarray, shape = (T, N, K)

    返回
    ----
    X_z : ndarray, shape = (T, N, K)
    """
    if X.ndim != 3:
        raise ValueError(f"X 应为三维数组 (T, N, K)，当前维度为 {X.ndim}")

    X_z = np.zeros_like(X, dtype=float)
    for t in range(X.shape[0]):
        X_z[t] = zscore_cross_section(X[t])
    return X_z


def zscore_global(X: np.ndarray) -> np.ndarray:
    """
    对整个面板做全局标准化（按每个因子维度统一标准化）

    参数
    ----
    X : ndarray, shape = (T, N, K)

    返回
    ----
    X_z : ndarray, shape = (T, N, K)
    """
    if X.ndim != 3:
        raise ValueError(f"X 应为三维数组 (T, N, K)，当前维度为 {X.ndim}")

    mean = np.mean(X, axis=(0, 1), keepdims=True)
    std = _safe_std(X, axis=(0, 1), keepdims=True)
    X_z = (X - mean) / std
    return X_z


def winsorize_cross_section(
    X_t: np.ndarray,
    lower: float = 0.01,
    upper: float = 0.99,
) -> np.ndarray:
    """
    对单期因子做横截面分位数缩尾

    参数
    ----
    X_t : ndarray, shape = (N, K)
        单期因子矩阵
    lower : float
        下分位数，例如 0.01
    upper : float
        上分位数，例如 0.99

    返回
    ----
    X_t_w : ndarray, shape = (N, K)
        缩尾后的单期因子
    """
    if X_t.ndim != 2:
        raise ValueError(f"X_t 应为二维数组 (N, K)，当前维度为 {X_t.ndim}")
    if not (0.0 <= lower < upper <= 1.0):
        raise ValueError("lower 和 upper 必须满足 0 <= lower < upper <= 1")

    lower_q = np.quantile(X_t, lower, axis=0, keepdims=True)
    upper_q = np.quantile(X_t, upper, axis=0, keepdims=True)
    X_t_w = np.clip(X_t, lower_q, upper_q)
    return X_t_w


def winsorize_panel(
    X: np.ndarray,
    lower: float = 0.01,
    upper: float = 0.99,
) -> np.ndarray:
    """
    对整个面板逐期做横截面缩尾

    参数
    ----
    X : ndarray, shape = (T, N, K)
    lower : float
        下分位数
    upper : float
        上分位数

    返回
    ----
    X_w : ndarray, shape = (T, N, K)
    """
    if X.ndim != 3:
        raise ValueError(f"X 应为三维数组 (T, N, K)，当前维度为 {X.ndim}")

    X_w = np.zeros_like(X, dtype=float)
    for t in range(X.shape[0]):
        X_w[t] = winsorize_cross_section(X[t], lower=lower, upper=upper)
    return X_w


def fillna_panel(X: np.ndarray, method: str = "zero") -> np.ndarray:
    """
    对面板数据中的缺失值进行填充

    支持方法：
    - "zero"  : 用 0 填充
    - "mean"  : 用每个因子的全局均值填充

    参数
    ----
    X : ndarray, shape = (T, N, K)
    method : str
        填充方式

    返回
    ----
    X_filled : ndarray, shape = (T, N, K)
    """
    if X.ndim != 3:
        raise ValueError(f"X 应为三维数组 (T, N, K)，当前维度为 {X.ndim}")

    X_filled = X.copy()

    if method == "zero":
        X_filled = np.nan_to_num(X_filled, nan=0.0)
        return X_filled

    if method == "mean":
        factor_mean = np.nanmean(X_filled, axis=(0, 1), keepdims=True)
        nan_mask = np.isnan(X_filled)
        X_filled[nan_mask] = np.take(
            factor_mean.reshape(-1),
            np.where(nan_mask)[2]
        )
        return X_filled

    raise ValueError("method 仅支持 'zero' 或 'mean'")


def preprocess_panel(
    X: np.ndarray,
    fillna_method: str = "zero",
    do_winsorize: bool = True,
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
    do_zscore: bool = True,
    zscore_method: str = "cross_section",
) -> np.ndarray:
    """
    统一预处理接口

    处理顺序：
    1. 缺失值填充
    2. 缩尾
    3. 标准化

    参数
    ----
    X : ndarray, shape = (T, N, K)
    fillna_method : str
        缺失值填充方式，"zero" 或 "mean"
    do_winsorize : bool
        是否缩尾
    winsor_lower : float
        缩尾下分位数
    winsor_upper : float
        缩尾上分位数
    do_zscore : bool
        是否标准化
    zscore_method : str
        "cross_section" 或 "global"

    返回
    ----
    X_out : ndarray, shape = (T, N, K)
        预处理后的面板数据
    """
    X_out = fillna_panel(X, method=fillna_method)

    if do_winsorize:
        X_out = winsorize_panel(X_out, lower=winsor_lower, upper=winsor_upper)

    if do_zscore:
        if zscore_method == "cross_section":
            X_out = zscore_panel(X_out)
        elif zscore_method == "global":
            X_out = zscore_global(X_out)
        else:
            raise ValueError("zscore_method 仅支持 'cross_section' 或 'global'")

    return X_out