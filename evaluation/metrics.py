"""
模型评估指标模块

包含：
1. MSE
2. MAE
3. 样本外 R^2
4. 单期 IC / RankIC
5. 面板平均 IC / RankIC
6. 统一评估接口
"""

from typing import Dict, List
import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算均方误差 MSE

    参数
    ----
    y_true : ndarray
    y_pred : ndarray

    返回
    ----
    float
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true 和 y_pred 的形状必须一致")

    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算平均绝对误差 MAE
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true 和 y_pred 的形状必须一致")

    return float(np.mean(np.abs(y_true - y_pred)))


def r2_oos(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算样本外 R^2

    公式：
        R^2 = 1 - SSE / SST

    其中：
        SSE = sum((y_true - y_pred)^2)
        SST = sum((y_true - mean(y_true))^2)

    参数
    ----
    y_true : ndarray
    y_pred : ndarray

    返回
    ----
    float
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true 和 y_pred 的形状必须一致")

    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)

    if sst < 1e-12:
        return 0.0

    return float(1.0 - sse / sst)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    安全计算 Pearson 相关系数
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)

    if x.shape[0] != y.shape[0]:
        raise ValueError("x 和 y 的长度必须一致")

    x = x - np.mean(x)
    y = y - np.mean(y)

    denom = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
    if denom < 1e-12:
        return 0.0

    return float(np.sum(x * y) / denom)


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """
    计算平均秩（处理并列值）

    不依赖 scipy，便于项目可移植
    """
    x = np.asarray(x).reshape(-1)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)

    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1

        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    return ranks


def information_coefficient(y_true_t: np.ndarray, y_pred_t: np.ndarray) -> float:
    """
    计算单期 IC（Pearson 截面相关）

    参数
    ----
    y_true_t : ndarray, shape = (N,)
    y_pred_t : ndarray, shape = (N,)

    返回
    ----
    float
    """
    return _safe_corr(y_true_t, y_pred_t)


def rank_ic(y_true_t: np.ndarray, y_pred_t: np.ndarray) -> float:
    """
    计算单期 RankIC（Spearman 截面相关）

    参数
    ----
    y_true_t : ndarray, shape = (N,)
    y_pred_t : ndarray, shape = (N,)

    返回
    ----
    float
    """
    rank_true = _rankdata_average(y_true_t)
    rank_pred = _rankdata_average(y_pred_t)
    return _safe_corr(rank_true, rank_pred)


def panel_ic(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    逐期计算 IC 序列

    参数
    ----
    y_true : ndarray, shape = (T, N)
    y_pred : ndarray, shape = (T, N)

    返回
    ----
    ic_series : ndarray, shape = (T,)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true 和 y_pred 的形状必须一致")
    if y_true.ndim != 2:
        raise ValueError("panel_ic 要求输入为二维数组 (T, N)")

    values: List[float] = []
    for t in range(y_true.shape[0]):
        values.append(information_coefficient(y_true[t], y_pred[t]))

    return np.array(values, dtype=float)


def panel_rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    逐期计算 RankIC 序列

    参数
    ----
    y_true : ndarray, shape = (T, N)
    y_pred : ndarray, shape = (T, N)

    返回
    ----
    rank_ic_series : ndarray, shape = (T,)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true 和 y_pred 的形状必须一致")
    if y_true.ndim != 2:
        raise ValueError("panel_rank_ic 要求输入为二维数组 (T, N)")

    values: List[float] = []
    for t in range(y_true.shape[0]):
        values.append(rank_ic(y_true[t], y_pred[t]))

    return np.array(values, dtype=float)


def ic_ir(ic_series: np.ndarray) -> float:
    """
    计算 ICIR

    公式：
        ICIR = mean(IC) / std(IC)

    参数
    ----
    ic_series : ndarray, shape = (T,)

    返回
    ----
    float
    """
    ic_series = np.asarray(ic_series).reshape(-1)

    mean_ic = np.mean(ic_series)
    std_ic = np.std(ic_series)

    if std_ic < 1e-12:
        return 0.0

    return float(mean_ic / std_ic)


def evaluate_panel_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    对面板预测结果进行统一评估

    参数
    ----
    y_true : ndarray, shape = (T, N)
    y_pred : ndarray, shape = (T, N)

    返回
    ----
    result : dict
        {
            "mse": ...,
            "mae": ...,
            "r2": ...,
            "mean_ic": ...,
            "ic_ir": ...,
            "mean_rank_ic": ...,
            "rank_ic_ir": ...
        }
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true 和 y_pred 的形状必须一致")
    if y_true.ndim != 2:
        raise ValueError("evaluate_panel_predictions 要求输入为二维数组 (T, N)")

    ic_series = panel_ic(y_true, y_pred)
    rank_ic_series = panel_rank_ic(y_true, y_pred)

    result = {
        "mse": mse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2_oos(y_true, y_pred),
        "mean_ic": float(np.mean(ic_series)),
        "ic_ir": ic_ir(ic_series),
        "mean_rank_ic": float(np.mean(rank_ic_series)),
        "rank_ic_ir": ic_ir(rank_ic_series),
    }
    return result