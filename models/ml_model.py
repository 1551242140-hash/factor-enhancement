"""
机器学习模型模块（可选）

当前提供：
1. Random Forest 回归
2. XGBoost 回归（若环境未安装 xgboost，则自动跳过）

说明：
- 该文件是可选扩展，不影响主实验主线
- 主实验建议先跑 linear_model.py 和 gnn_model.py
"""

from typing import Dict, Optional
import numpy as np

from sklearn.ensemble import RandomForestRegressor


def _to_2d_features(X: np.ndarray) -> np.ndarray:
    """
    将输入特征统一转换为二维数组
    """
    if X.ndim == 2:
        return X
    if X.ndim == 3:
        t_periods, n_stocks, n_features = X.shape
        return X.reshape(t_periods * n_stocks, n_features)
    raise ValueError(f"X 仅支持二维或三维数组，当前维度为 {X.ndim}")


def _to_1d_target(y: np.ndarray) -> np.ndarray:
    """
    将目标变量统一转换为一维数组
    """
    if y.ndim == 1:
        return y
    if y.ndim == 2:
        return y.reshape(-1)
    raise ValueError(f"y 仅支持一维或二维数组，当前维度为 {y.ndim}")


class RandomForestFactorModel:
    """
    随机森林回归模型
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_2d = _to_2d_features(X)
        y_1d = _to_1d_target(y)
        self.model.fit(X_2d, y_1d)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:
            return self.model.predict(X)

        if X.ndim == 3:
            t_periods, n_stocks, _ = X.shape
            X_2d = X.reshape(t_periods * n_stocks, -1)
            y_pred = self.model.predict(X_2d)
            return y_pred.reshape(t_periods, n_stocks)

        raise ValueError(f"X 仅支持二维或三维数组，当前维度为 {X.ndim}")


class XGBoostFactorModel:
    """
    XGBoost 回归模型
    若环境未安装 xgboost，则初始化时报错
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "当前环境未安装 xgboost。若要使用 XGBoostFactorModel，请先安装 xgboost。"
            ) from exc

        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            objective="reg:squarederror",
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_2d = _to_2d_features(X)
        y_1d = _to_1d_target(y)
        self.model.fit(X_2d, y_1d)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:
            return self.model.predict(X)

        if X.ndim == 3:
            t_periods, n_stocks, _ = X.shape
            X_2d = X.reshape(t_periods * n_stocks, -1)
            y_pred = self.model.predict(X_2d)
            return y_pred.reshape(t_periods, n_stocks)

        raise ValueError(f"X 仅支持二维或三维数组，当前维度为 {X.ndim}")


def run_ml_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    model_type: str = "rf",
) -> Dict[str, np.ndarray]:
    """
    统一运行机器学习 baseline

    参数
    ----
    model_type : str
        "rf" 或 "xgb"
    """
    if model_type == "rf":
        model = RandomForestFactorModel()
    elif model_type == "xgb":
        model = XGBoostFactorModel()
    else:
        raise ValueError("model_type 仅支持 'rf' 或 'xgb'")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "model": model,
        "y_pred": y_pred,
    }