"""
生成模拟股票因子面板数据

目标：
1. 生成 T 个时点、N 只股票、K 个因子的面板数据 X
2. 股票被划分到若干潜在组（类似行业/主题）
3. 同组股票共享组层面的因子中心，从而形成“相似股票结构”
"""

from typing import Dict, Any
import numpy as np


def simulate_factor_panel(
    n_stocks: int,
    t_periods: int,
    k_features: int,
    n_groups: int,
    rho_group: float,
    sigma_group: float,
    sigma_idio: float,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    生成带有潜在分组结构的股票因子面板数据

    参数
    ----
    n_stocks : int
        股票数量 N
    t_periods : int
        时间长度 T
    k_features : int
        因子数量 K
    n_groups : int
        潜在分组数量 G
    rho_group : float
        组因子的时间持续性，越接近1表示组中心变化越平滑
    sigma_group : float
        组层面随机扰动强度
    sigma_idio : float
        个股层面噪声强度
    seed : int
        随机种子

    返回
    ----
    result : dict
        {
            "X": ndarray, shape = (T, N, K)
                股票因子面板数据
            "group_ids": ndarray, shape = (N,)
                每只股票所属组编号
            "group_centers": ndarray, shape = (T, G, K)
                每个时点、每个组的因子中心
        }
    """
    rng = np.random.default_rng(seed)

    # -------------------------
    # 1. 给每只股票分配潜在组
    # -------------------------
    # 尽量均匀分组；如果 N 不能整除 G，后面剩余的股票随机分配
    base_group_ids = np.repeat(np.arange(n_groups), n_stocks // n_groups)
    remainder = n_stocks - len(base_group_ids)

    if remainder > 0:
        extra_group_ids = rng.choice(np.arange(n_groups), size=remainder, replace=True)
        group_ids = np.concatenate([base_group_ids, extra_group_ids])
    else:
        group_ids = base_group_ids.copy()

    rng.shuffle(group_ids)

    # -------------------------
    # 2. 生成组层面的因子中心
    #    group_centers[t, g, :]
    # -------------------------
    group_centers = np.zeros((t_periods, n_groups, k_features), dtype=float)

    # 初始时点
    group_centers[0] = rng.normal(
        loc=0.0,
        scale=sigma_group,
        size=(n_groups, k_features),
    )

    # 时间递推：AR(1) 过程
    for t in range(1, t_periods):
        shock_t = rng.normal(
            loc=0.0,
            scale=sigma_group,
            size=(n_groups, k_features),
        )
        group_centers[t] = rho_group * group_centers[t - 1] + shock_t

    # -------------------------
    # 3. 生成股票因子 X
    #    X[t, i, :] = 组中心 + 个股噪声
    # -------------------------
    X = np.zeros((t_periods, n_stocks, k_features), dtype=float)

    for t in range(t_periods):
        for i in range(n_stocks):
            g = group_ids[i]
            idio_noise = rng.normal(loc=0.0, scale=sigma_idio, size=k_features)
            X[t, i, :] = group_centers[t, g, :] + idio_noise

    return {
        "X": X,
        "group_ids": group_ids,
        "group_centers": group_centers,
    }


def simulate_hidden_signal_from_groups(
    group_ids: np.ndarray,
    t_periods: int,
    n_groups: int,
    rho_hidden: float = 0.8,
    sigma_hidden: float = 1.0,
    sigma_idio: float = 0.3,
    seed: int = 42,
) -> np.ndarray:
    """
    生成一个“隐藏信号” z[t, i]，可用于 graph_misaligned 场景。
    例如：收益真实依赖隐藏主题，但构图时没有使用这个隐藏主题。

    参数
    ----
    group_ids : ndarray, shape = (N,)
        股票所属组
    t_periods : int
        时间长度 T
    n_groups : int
        分组数 G
    rho_hidden : float
        隐藏组信号的时间持续性
    sigma_hidden : float
        隐藏组信号波动
    sigma_idio : float
        个股层面噪声
    seed : int
        随机种子

    返回
    ----
    hidden_signal : ndarray, shape = (T, N)
        每个时点每只股票的隐藏信号
    """
    rng = np.random.default_rng(seed)

    hidden_group = np.zeros((t_periods, n_groups), dtype=float)
    hidden_group[0] = rng.normal(loc=0.0, scale=sigma_hidden, size=n_groups)

    for t in range(1, t_periods):
        shock_t = rng.normal(loc=0.0, scale=sigma_hidden, size=n_groups)
        hidden_group[t] = rho_hidden * hidden_group[t - 1] + shock_t

    n_stocks = len(group_ids)
    hidden_signal = np.zeros((t_periods, n_stocks), dtype=float)

    for t in range(t_periods):
        for i in range(n_stocks):
            g = group_ids[i]
            hidden_signal[t, i] = hidden_group[t, g] + rng.normal(0.0, sigma_idio)

    return hidden_signal