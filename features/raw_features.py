"""
原始因子特征处理模块

作用：
1. 提供“只使用自身因子”的 baseline 特征
2. 与 graph_features.py 形成对照（是否使用邻居信息）
3. 提供统一接口，方便 main.py 调用
"""

from typing import Dict
import numpy as np


def get_raw_features(X: np.ndarray) -> np.ndarray:
    """
    直接返回原始因子（不做图增强）

    参数
    ----
    X : ndarray, shape = (T, N, K)
        面板因子数据

    返回
    ----
    X_raw : ndarray, shape = (T, N, K)
        原始因子（直接复制，避免引用问题）
    """
    if X.ndim != 3:
        raise ValueError(f"X 应为三维数组 (T, N, K)，当前维度为 {X.ndim}")

    return X.copy()


def build_lagged_features(X: np.ndarray, lag: int = 1) -> np.ndarray:
    """
    构造滞后因子（用于更贴近真实预测问题）

    例如：
        用 t-1 的因子预测 t 的收益

    参数
    ----
    X : ndarray, shape = (T, N, K)
    lag : int
        滞后阶数

    返回
    ----
    X_lagged : ndarray, shape = (T-lag, N, K)
    """
    if lag <= 0:
        raise ValueError("lag 必须大于 0")

    if X.shape[0] <= lag:
        raise ValueError("时间长度不足以构造滞后特征")

    return X[:-lag]


def align_features_and_target(
    X: np.ndarray,
    y: np.ndarray,
    lag: int = 1
) -> Dict[str, np.ndarray]:
    """
    对齐因子与未来收益

    核心逻辑：
        X[t] -> y[t+1]

    参数
    ----
    X : ndarray, shape = (T, N, K)
    y : ndarray, shape = (T, N)
    lag : int
        滞后期（通常为1）

    返回
    ----
    dict:
        {
            "X_aligned": (T-lag, N, K),
            "y_aligned": (T-lag, N)
        }
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("X 和 y 的时间维度必须一致")

    X_aligned = X[:-lag]
    y_aligned = y[lag:]

    return {
        "X_aligned": X_aligned,
        "y_aligned": y_aligned
    }


def flatten_panel(
    X: np.ndarray,
    y: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    将面板数据展开为机器学习输入格式

    (T, N, K) -> (T*N, K)
    (T, N)    -> (T*N,)

    参数
    ----
    X : ndarray, shape = (T, N, K)
    y : ndarray, shape = (T, N)

    返回
    ----
    dict:
        {
            "X_flat": (T*N, K),
            "y_flat": (T*N,)
        }
    """
    T, N, K = X.shape

    X_flat = X.reshape(T * N, K)
    y_flat = y.reshape(T * N)

    return {
        "X_flat": X_flat,
        "y_flat": y_flat
    }


def build_raw_dataset(
    X: np.ndarray,
    y: np.ndarray,
    lag: int = 1,
    flatten: bool = True
) -> Dict[str, np.ndarray]:
    """
    一站式构建“原始因子数据集”

    流程：
        1. 对齐 X 和 y（滞后）
        2. 是否展开为 2D 数据

    参数
    ----
    X : ndarray, shape = (T, N, K)
    y : ndarray, shape = (T, N)
    lag : int
        滞后期
    flatten : bool
        是否展开

    返回
    ----
    dict:
        如果 flatten=True:
            {
                "X": (T*N, K),
                "y": (T*N,)
            }
        否则：
            {
                "X": (T, N, K),
                "y": (T, N)
            }
    """
    aligned = align_features_and_target(X, y, lag=lag)
    X_a = aligned["X_aligned"]
    y_a = aligned["y_aligned"]

    if flatten:
        flat = flatten_panel(X_a, y_a)
        return {
            "X": flat["X_flat"],
            "y": flat["y_flat"]
        }

    return {
        "X": X_a,
        "y": y_a
    }