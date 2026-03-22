"""
图矩阵工具模块

支持：
1. 加自环
2. 行归一化
3. 对称归一化（GCN 常用）
4. 计算度矩阵
5. 计算图拉普拉斯矩阵
6. 邻接矩阵对称化
"""

import numpy as np


def add_self_loops(A: np.ndarray, fill_value: float = 1.0) -> np.ndarray:
    """
    给邻接矩阵加入自环

    参数
    ----
    A : ndarray, shape = (N, N)
        邻接矩阵
    fill_value : float
        对角线填充值

    返回
    ----
    A_out : ndarray, shape = (N, N)
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A 必须是方阵")

    A_out = A.copy()
    np.fill_diagonal(A_out, np.diag(A_out) + fill_value)
    return A_out


def compute_degree_vector(A: np.ndarray) -> np.ndarray:
    """
    计算度向量 d_i = sum_j A_ij

    参数
    ----
    A : ndarray, shape = (N, N)

    返回
    ----
    degree : ndarray, shape = (N,)
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A 必须是方阵")
    return np.sum(A, axis=1)


def compute_degree_matrix(A: np.ndarray) -> np.ndarray:
    """
    计算度矩阵 D

    参数
    ----
    A : ndarray, shape = (N, N)

    返回
    ----
    D : ndarray, shape = (N, N)
    """
    degree = compute_degree_vector(A)
    return np.diag(degree)


def row_normalize(A: np.ndarray) -> np.ndarray:
    """
    行归一化：A_row = D^{-1} A

    作用：
    - 每一行和为 1
    - 可解释为“邻居特征加权平均”

    参数
    ----
    A : ndarray, shape = (N, N)

    返回
    ----
    A_norm : ndarray, shape = (N, N)
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A 必须是方阵")

    degree = np.sum(A, axis=1, keepdims=True)
    degree = np.where(degree < 1e-12, 1.0, degree)
    A_norm = A / degree
    return A_norm


def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """
    GCN 常用对称归一化：
        A_norm = D^{-1/2} A D^{-1/2}

    参数
    ----
    A : ndarray, shape = (N, N)

    返回
    ----
    A_norm : ndarray, shape = (N, N)
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A 必须是方阵")

    degree = np.sum(A, axis=1)
    degree = np.where(degree < 1e-12, 1.0, degree)
    d_inv_sqrt = 1.0 / np.sqrt(degree)
    D_inv_sqrt = np.diag(d_inv_sqrt)

    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm


def symmetrize_adjacency(A: np.ndarray, method: str = "mean") -> np.ndarray:
    """
    对邻接矩阵进行对称化

    参数
    ----
    A : ndarray, shape = (N, N)
    method : str
        "mean" -> 0.5 * (A + A.T)
        "max"  -> max(A, A.T)

    返回
    ----
    A_sym : ndarray, shape = (N, N)
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A 必须是方阵")

    if method == "mean":
        return 0.5 * (A + A.T)
    if method == "max":
        return np.maximum(A, A.T)

    raise ValueError("method 仅支持 'mean' 或 'max'")


def compute_laplacian(A: np.ndarray, normalized: bool = True) -> np.ndarray:
    """
    计算图拉普拉斯矩阵

    若 normalized=False:
        L = D - A

    若 normalized=True:
        L = I - D^{-1/2} A D^{-1/2}

    参数
    ----
    A : ndarray, shape = (N, N)
        邻接矩阵
    normalized : bool
        是否计算归一化拉普拉斯

    返回
    ----
    L : ndarray, shape = (N, N)
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A 必须是方阵")

    n = A.shape[0]

    if normalized:
        A_norm = normalize_adjacency(A)
        L = np.eye(n) - A_norm
    else:
        D = compute_degree_matrix(A)
        L = D - A

    return L


def ensure_nonnegative(A: np.ndarray) -> np.ndarray:
    """
    将邻接矩阵中的负值截断为 0

    参数
    ----
    A : ndarray

    返回
    ----
    A_out : ndarray
    """
    return np.maximum(A, 0.0)