"""
按时间维度切分训练集 / 验证集 / 测试集

说明：
- 这里是时间序列面板数据，所以必须按时间切，不能随机打乱
- 适用于 X: (T, N, K), y: (T, N), A: (T, N, N)
"""

from typing import Dict, Any, Optional
import numpy as np


def _validate_ratios(train_ratio: float, valid_ratio: float, test_ratio: float) -> None:
    """
    检查划分比例是否合法
    """
    total = train_ratio + valid_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"train_ratio + valid_ratio + test_ratio 必须等于 1，当前为 {total}"
        )
    if min(train_ratio, valid_ratio, test_ratio) <= 0:
        raise ValueError("三个划分比例都必须大于 0")


def time_series_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.6,
    valid_ratio: float = 0.2,
    test_ratio: float = 0.2,
    A: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    按时间维度切分面板数据

    参数
    ----
    X : ndarray, shape = (T, N, K)
        因子面板数据
    y : ndarray, shape = (T, N)
        收益数据
    train_ratio : float
        训练集比例
    valid_ratio : float
        验证集比例
    test_ratio : float
        测试集比例
    A : ndarray, shape = (T, N, N), optional
        图邻接矩阵面板，可选

    返回
    ----
    result : dict
        {
            "X_train", "X_valid", "X_test",
            "y_train", "y_valid", "y_test",
            "A_train", "A_valid", "A_test",   # 如果传入 A 则返回
            "idx_train", "idx_valid", "idx_test"
        }
    """
    _validate_ratios(train_ratio, valid_ratio, test_ratio)

    if X.ndim != 3:
        raise ValueError(f"X 应为三维数组 (T, N, K)，当前维度为 {X.ndim}")
    if y.ndim != 2:
        raise ValueError(f"y 应为二维数组 (T, N)，当前维度为 {y.ndim}")

    t_periods = X.shape[0]

    if y.shape[0] != t_periods:
        raise ValueError("X 和 y 的时间长度 T 必须一致")
    if X.shape[1] != y.shape[1]:
        raise ValueError("X 和 y 的股票数量 N 必须一致")

    if A is not None:
        if A.ndim != 3:
            raise ValueError(f"A 应为三维数组 (T, N, N)，当前维度为 {A.ndim}")
        if A.shape[0] != t_periods:
            raise ValueError("A 和 X 的时间长度 T 必须一致")
        if A.shape[1] != X.shape[1] or A.shape[2] != X.shape[1]:
            raise ValueError("A 的股票维度必须与 X 的股票数量一致")

    # -------------------------
    # 计算切分点
    # -------------------------
    train_end = int(t_periods * train_ratio)
    valid_end = int(t_periods * (train_ratio + valid_ratio))

    # 防止某段为空
    if train_end <= 0 or valid_end <= train_end or valid_end >= t_periods:
        raise ValueError(
            "时间切分失败，请检查 T 是否太小或切分比例是否不合理"
        )

    idx_train = np.arange(0, train_end)
    idx_valid = np.arange(train_end, valid_end)
    idx_test = np.arange(valid_end, t_periods)

    result = {
        "X_train": X[idx_train],
        "X_valid": X[idx_valid],
        "X_test": X[idx_test],
        "y_train": y[idx_train],
        "y_valid": y[idx_valid],
        "y_test": y[idx_test],
        "idx_train": idx_train,
        "idx_valid": idx_valid,
        "idx_test": idx_test,
    }

    if A is not None:
        result["A_train"] = A[idx_train]
        result["A_valid"] = A[idx_valid]
        result["A_test"] = A[idx_test]

    return result