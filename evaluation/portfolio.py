"""
组合评估模块

包含：
1. 单期分组收益
2. 面板分组收益
3. 多空组合收益
4. Sharpe 比率
5. 累积收益序列

说明：
- 默认按预测值 y_pred 横截面排序
- 再用真实收益 y_true 计算各组未来收益
"""

from typing import Dict, List
import numpy as np


def _get_quantile_bins(scores: np.ndarray, n_bins: int) -> np.ndarray:
    """
    根据分数将样本分到 n_bins 个组

    返回
    ----
    bin_idx : ndarray, shape = (N,)
        取值为 0,1,...,n_bins-1
    """
    scores = np.asarray(scores).reshape(-1)
    n = len(scores)

    if n_bins <= 1:
        raise ValueError("n_bins 必须大于 1")
    if n < n_bins:
        raise ValueError("样本数必须大于等于分组数")

    order = np.argsort(scores)
    bin_idx = np.empty(n, dtype=int)

    for rank, idx in enumerate(order):
        group = min(rank * n_bins // n, n_bins - 1)
        bin_idx[idx] = group

    return bin_idx


def quantile_portfolio_returns(
    y_true_t: np.ndarray,
    y_pred_t: np.ndarray,
    n_bins: int = 5,
) -> np.ndarray:
    """
    计算单期分组组合收益

    参数
    ----
    y_true_t : ndarray, shape = (N,)
        单期真实未来收益
    y_pred_t : ndarray, shape = (N,)
        单期预测值（排序依据）
    n_bins : int
        分组数量

    返回
    ----
    group_returns : ndarray, shape = (n_bins,)
        每组平均收益
    """
    y_true_t = np.asarray(y_true_t).reshape(-1)
    y_pred_t = np.asarray(y_pred_t).reshape(-1)

    if y_true_t.shape[0] != y_pred_t.shape[0]:
        raise ValueError("y_true_t 和 y_pred_t 的长度必须一致")

    groups = _get_quantile_bins(y_pred_t, n_bins=n_bins)

    group_returns = np.zeros(n_bins, dtype=float)
    for g in range(n_bins):
        mask = groups == g
        if np.sum(mask) == 0:
            group_returns[g] = 0.0
        else:
            group_returns[g] = float(np.mean(y_true_t[mask]))

    return group_returns


def panel_quantile_portfolio_returns(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 5,
) -> np.ndarray:
    """
    逐期计算分组组合收益

    参数
    ----
    y_true : ndarray, shape = (T, N)
    y_pred : ndarray, shape = (T, N)
    n_bins : int
        分组数

    返回
    ----
    group_returns_panel : ndarray, shape = (T, n_bins)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true 和 y_pred 的形状必须一致")
    if y_true.ndim != 2:
        raise ValueError("panel_quantile_portfolio_returns 要求输入为二维数组 (T, N)")

    out: List[np.ndarray] = []
    for t in range(y_true.shape[0]):
        out.append(quantile_portfolio_returns(y_true[t], y_pred[t], n_bins=n_bins))

    return np.stack(out, axis=0)


def long_short_return(
    y_true_t: np.ndarray,
    y_pred_t: np.ndarray,
    top_quantile: float = 0.2,
    bottom_quantile: float = 0.2,
) -> float:
    """
    计算单期多空组合收益

    做法：
    - 做多预测值最高的 top_quantile 比例股票
    - 做空预测值最低的 bottom_quantile 比例股票
    - 返回 long - short

    参数
    ----
    y_true_t : ndarray, shape = (N,)
    y_pred_t : ndarray, shape = (N,)
    top_quantile : float
        多头比例
    bottom_quantile : float
        空头比例

    返回
    ----
    ls_ret : float
    """
    y_true_t = np.asarray(y_true_t).reshape(-1)
    y_pred_t = np.asarray(y_pred_t).reshape(-1)

    if y_true_t.shape[0] != y_pred_t.shape[0]:
        raise ValueError("y_true_t 和 y_pred_t 的长度必须一致")

    n = len(y_true_t)
    n_top = max(1, int(np.floor(n * top_quantile)))
    n_bottom = max(1, int(np.floor(n * bottom_quantile)))

    order = np.argsort(y_pred_t)

    short_idx = order[:n_bottom]
    long_idx = order[-n_top:]

    long_ret = float(np.mean(y_true_t[long_idx]))
    short_ret = float(np.mean(y_true_t[short_idx]))

    return long_ret - short_ret


def panel_long_short_returns(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_quantile: float = 0.2,
    bottom_quantile: float = 0.2,
) -> np.ndarray:
    """
    逐期计算多空组合收益序列

    参数
    ----
    y_true : ndarray, shape = (T, N)
    y_pred : ndarray, shape = (T, N)

    返回
    ----
    ls_series : ndarray, shape = (T,)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true 和 y_pred 的形状必须一致")
    if y_true.ndim != 2:
        raise ValueError("panel_long_short_returns 要求输入为二维数组 (T, N)")

    values = []
    for t in range(y_true.shape[0]):
        values.append(
            long_short_return(
                y_true[t],
                y_pred[t],
                top_quantile=top_quantile,
                bottom_quantile=bottom_quantile,
            )
        )

    return np.array(values, dtype=float)


def sharpe_ratio(
    returns: np.ndarray,
    annualize: bool = False,
    periods_per_year: int = 12,
) -> float:
    """
    计算 Sharpe 比率（无风险利率默认为0）

    参数
    ----
    returns : ndarray
        收益序列
    annualize : bool
        是否年化
    periods_per_year : int
        每年期数（如月频=12，日频=252）

    返回
    ----
    float
    """
    returns = np.asarray(returns).reshape(-1)

    mean_ret = np.mean(returns)
    std_ret = np.std(returns)

    if std_ret < 1e-12:
        return 0.0

    sr = mean_ret / std_ret

    if annualize:
        sr *= np.sqrt(periods_per_year)

    return float(sr)


def cumulative_returns(returns: np.ndarray) -> np.ndarray:
    """
    计算累积收益曲线

    采用简单收益累乘：
        cum_t = prod(1 + r_s) - 1

    参数
    ----
    returns : ndarray, shape = (T,)

    返回
    ----
    cum : ndarray, shape = (T,)
    """
    returns = np.asarray(returns).reshape(-1)
    return np.cumprod(1.0 + returns) - 1.0


def evaluate_portfolio_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 5,
    top_quantile: float = 0.2,
    bottom_quantile: float = 0.2,
    annualize_sharpe: bool = False,
    periods_per_year: int = 12,
) -> Dict[str, object]:
    """
    统一评估组合表现

    返回
    ----
    result : dict
        {
            "group_returns_panel": (T, n_bins),
            "mean_group_returns": (n_bins,),
            "ls_returns": (T,),
            "mean_ls_return": float,
            "ls_sharpe": float,
            "ls_cum_returns": (T,)
        }
    """
    group_panel = panel_quantile_portfolio_returns(
        y_true=y_true,
        y_pred=y_pred,
        n_bins=n_bins,
    )

    ls_returns = panel_long_short_returns(
        y_true=y_true,
        y_pred=y_pred,
        top_quantile=top_quantile,
        bottom_quantile=bottom_quantile,
    )

    result = {
        "group_returns_panel": group_panel,
        "mean_group_returns": np.mean(group_panel, axis=0),
        "ls_returns": ls_returns,
        "mean_ls_return": float(np.mean(ls_returns)),
        "ls_sharpe": sharpe_ratio(
            ls_returns,
            annualize=annualize_sharpe,
            periods_per_year=periods_per_year,
        ),
        "ls_cum_returns": cumulative_returns(ls_returns),
    }
    return result