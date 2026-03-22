"""
生成未来收益

支持三种场景：
1. self_only：收益只由自身因子决定，图无效
2. graph_helpful：收益同时由自身因子和邻居因子决定，图有效
3. graph_misaligned：收益依赖隐藏信号，但构图依据与真实机制错配，图可能无效或有害
"""

from typing import Optional
import numpy as np


def _check_beta_shape(beta: np.ndarray, k_features: int, name: str) -> None:
    """
    检查 beta 维度是否与因子数量匹配
    """
    if beta.ndim != 1:
        raise ValueError(f"{name} 必须是一维向量，当前形状为 {beta.shape}")
    if beta.shape[0] != k_features:
        raise ValueError(
            f"{name} 长度必须等于因子数量 K={k_features}，当前长度为 {beta.shape[0]}"
        )


def generate_returns_self_only(
    X: np.ndarray,
    beta_self: np.ndarray,
    noise_std: float,
    seed: int = 42,
) -> np.ndarray:
    """
    场景1：收益只由自身因子决定

    数学形式：
        r[t, i] = x[t, i]^T beta_self + eps[t, i]

    参数
    ----
    X : ndarray, shape = (T, N, K)
        股票因子面板
    beta_self : ndarray, shape = (K,)
        自身因子系数
    noise_std : float
        噪声标准差
    seed : int
        随机种子

    返回
    ----
    y : ndarray, shape = (T, N)
        每期每只股票的未来收益
    """
    rng = np.random.default_rng(seed)
    t_periods, n_stocks, k_features = X.shape

    _check_beta_shape(beta_self, k_features, "beta_self")

    signal = np.einsum("tnk,k->tn", X, beta_self)
    noise = rng.normal(loc=0.0, scale=noise_std, size=(t_periods, n_stocks))
    y = signal + noise
    return y


def generate_returns_with_graph_signal(
    X: np.ndarray,
    A_true: np.ndarray,
    beta_self: np.ndarray,
    beta_neighbor: np.ndarray,
    gamma: float,
    noise_std: float,
    seed: int = 42,
) -> np.ndarray:
    """
    场景2：收益由自身因子 + 邻居因子共同决定（图有效）

    数学形式：
        r[t, i] = x[t, i]^T beta_self
                  + gamma * (A_true[t] X[t] beta_neighbor)_i
                  + eps[t, i]

    其中：
        A_true[t] 是第 t 期的“真实图”邻接矩阵（最好已做行归一化）

    参数
    ----
    X : ndarray, shape = (T, N, K)
        股票因子面板
    A_true : ndarray, shape = (T, N, N) 或 (N, N)
        真实图；若是静态图，则会对所有时点复用
    beta_self : ndarray, shape = (K,)
        自身因子系数
    beta_neighbor : ndarray, shape = (K,)
        邻居因子系数
    gamma : float
        邻居影响强度
    noise_std : float
        噪声标准差
    seed : int
        随机种子

    返回
    ----
    y : ndarray, shape = (T, N)
        每期每只股票的未来收益
    """
    rng = np.random.default_rng(seed)
    t_periods, n_stocks, k_features = X.shape

    _check_beta_shape(beta_self, k_features, "beta_self")
    _check_beta_shape(beta_neighbor, k_features, "beta_neighbor")

    if A_true.ndim == 2:
        A_true = np.repeat(A_true[None, :, :], t_periods, axis=0)

    if A_true.shape != (t_periods, n_stocks, n_stocks):
        raise ValueError(
            f"A_true 形状应为 {(t_periods, n_stocks, n_stocks)}，当前为 {A_true.shape}"
        )

    self_signal = np.einsum("tnk,k->tn", X, beta_self)

    # 先算每只股票的“邻居有效因子暴露”
    neighbor_raw = np.einsum("tnk,k->tn", X, beta_neighbor)

    # 再通过图传播，把邻居暴露聚合到每个节点
    neighbor_signal = np.zeros((t_periods, n_stocks), dtype=float)
    for t in range(t_periods):
        neighbor_signal[t] = A_true[t] @ neighbor_raw[t]

    noise = rng.normal(loc=0.0, scale=noise_std, size=(t_periods, n_stocks))
    y = self_signal + gamma * neighbor_signal + noise
    return y


def generate_returns_misaligned_graph(
    X: np.ndarray,
    hidden_signal: np.ndarray,
    beta_self: np.ndarray,
    delta_hidden: float,
    noise_std: float,
    seed: int = 42,
) -> np.ndarray:
    """
    场景3：收益真正依赖“隐藏信号”，而不是构图所依据的显式因子关系

    数学形式：
        r[t, i] = x[t, i]^T beta_self + delta_hidden * hidden_signal[t, i] + eps[t, i]

    这可用于模拟“图构建错配”的情况：
    例如你用因子相似图构图，但收益其实由另一个隐藏主题驱动。

    参数
    ----
    X : ndarray, shape = (T, N, K)
        股票因子面板
    hidden_signal : ndarray, shape = (T, N)
        隐藏收益驱动信号
    beta_self : ndarray, shape = (K,)
        自身因子系数
    delta_hidden : float
        隐藏信号强度
    noise_std : float
        噪声标准差
    seed : int
        随机种子

    返回
    ----
    y : ndarray, shape = (T, N)
        每期每只股票的未来收益
    """
    rng = np.random.default_rng(seed)
    t_periods, n_stocks, k_features = X.shape

    _check_beta_shape(beta_self, k_features, "beta_self")

    if hidden_signal.shape != (t_periods, n_stocks):
        raise ValueError(
            f"hidden_signal 形状应为 {(t_periods, n_stocks)}，当前为 {hidden_signal.shape}"
        )

    self_signal = np.einsum("tnk,k->tn", X, beta_self)
    noise = rng.normal(loc=0.0, scale=noise_std, size=(t_periods, n_stocks))
    y = self_signal + delta_hidden * hidden_signal + noise
    return y


def generate_returns_by_scenario(
    X: np.ndarray,
    scenario: str,
    beta_self: np.ndarray,
    noise_std: float,
    seed: int = 42,
    A_true: Optional[np.ndarray] = None,
    beta_neighbor: Optional[np.ndarray] = None,
    gamma: float = 0.0,
    hidden_signal: Optional[np.ndarray] = None,
    delta_hidden: float = 1.0,
) -> np.ndarray:
    """
    根据场景统一生成收益

    参数
    ----
    scenario : str
        "self_only" / "graph_helpful" / "graph_misaligned"

    返回
    ----
    y : ndarray, shape = (T, N)
    """
    if scenario == "self_only":
        return generate_returns_self_only(
            X=X,
            beta_self=beta_self,
            noise_std=noise_std,
            seed=seed,
        )

    if scenario == "graph_helpful":
        if A_true is None:
            raise ValueError("graph_helpful 场景下必须提供 A_true")
        if beta_neighbor is None:
            raise ValueError("graph_helpful 场景下必须提供 beta_neighbor")
        return generate_returns_with_graph_signal(
            X=X,
            A_true=A_true,
            beta_self=beta_self,
            beta_neighbor=beta_neighbor,
            gamma=gamma,
            noise_std=noise_std,
            seed=seed,
        )

    if scenario == "graph_misaligned":
        if hidden_signal is None:
            raise ValueError("graph_misaligned 场景下必须提供 hidden_signal")
        return generate_returns_misaligned_graph(
            X=X,
            hidden_signal=hidden_signal,
            beta_self=beta_self,
            delta_hidden=delta_hidden,
            noise_std=noise_std,
            seed=seed,
        )

    raise ValueError(
        f"不支持的 scenario={scenario}，可选值为 "
        f"'self_only', 'graph_helpful', 'graph_misaligned'"
    )