"""
线性模型模块

包含：
1. OLS 线性回归
2. Ridge 回归
3. 面板数据训练与预测接口

说明：
- 输入可以是二维特征 (n_samples, n_features)
- 也可以是三维面板特征 (T, N, K)，内部会自动展开
"""

from typing import Dict, Optional
import numpy as np


def _to_2d_features(X: np.ndarray) -> np.ndarray:
    """
    将输入特征统一转换为二维数组

    支持：
    - X: (n_samples, n_features)
    - X: (T, N, K)

    返回
    ----
    X_2d : ndarray, shape = (n_samples, n_features)
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

    支持：
    - y: (n_samples,)
    - y: (T, N)

    返回
    ----
    y_1d : ndarray, shape = (n_samples,)
    """
    if y.ndim == 1:
        return y
    if y.ndim == 2:
        return y.reshape(-1)
    raise ValueError(f"y 仅支持一维或二维数组，当前维度为 {y.ndim}")


class LinearFactorModel:
    """
    普通最小二乘（OLS）线性回归
    """

    def __init__(self, fit_intercept: bool = True):
        """
        参数
        ----
        fit_intercept : bool
            是否拟合截距项
        """
        self.fit_intercept = fit_intercept
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.is_fitted_: bool = False

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """
        给特征矩阵添加常数项
        """
        if not self.fit_intercept:
            return X
        ones = np.ones((X.shape[0], 1), dtype=float)
        return np.hstack([ones, X])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        拟合 OLS

        参数
        ----
        X : ndarray
            特征矩阵，支持 (n_samples, n_features) 或 (T, N, K)
        y : ndarray
            目标变量，支持 (n_samples,) 或 (T, N)
        """
        X_2d = _to_2d_features(X)
        y_1d = _to_1d_target(y)

        if X_2d.shape[0] != y_1d.shape[0]:
            raise ValueError("X 和 y 的样本数不一致")

        X_design = self._add_intercept(X_2d)

        # 最小二乘闭式解
        beta_hat, _, _, _ = np.linalg.lstsq(X_design, y_1d, rcond=None)

        if self.fit_intercept:
            self.intercept_ = float(beta_hat[0])
            self.coef_ = beta_hat[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta_hat

        self.is_fitted_ = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测

        参数
        ----
        X : ndarray
            支持二维或三维输入

        返回
        ----
        y_pred : ndarray
            若输入是二维，输出 shape = (n_samples,)
            若输入是三维，输出 shape = (T, N)
        """
        if not self.is_fitted_:
            raise RuntimeError("模型尚未拟合，请先调用 fit")

        if X.ndim == 2:
            return X @ self.coef_ + self.intercept_

        if X.ndim == 3:
            t_periods, n_stocks, _ = X.shape
            X_2d = X.reshape(t_periods * n_stocks, -1)
            y_pred = X_2d @ self.coef_ + self.intercept_
            return y_pred.reshape(t_periods, n_stocks)

        raise ValueError(f"X 仅支持二维或三维数组，当前维度为 {X.ndim}")

    def fit_predict(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        """
        拟合后直接预测
        """
        self.fit(X_train, y_train)
        return self.predict(X_test)


class RidgeFactorModel:
    """
    Ridge 回归
    """

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        """
        参数
        ----
        alpha : float
            L2 正则强度
        fit_intercept : bool
            是否拟合截距项
        """
        if alpha < 0:
            raise ValueError("alpha 必须大于等于 0")

        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.is_fitted_: bool = False

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """
        给特征矩阵添加常数项
        """
        if not self.fit_intercept:
            return X
        ones = np.ones((X.shape[0], 1), dtype=float)
        return np.hstack([ones, X])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        拟合 Ridge 回归
        """
        X_2d = _to_2d_features(X)
        y_1d = _to_1d_target(y)

        if X_2d.shape[0] != y_1d.shape[0]:
            raise ValueError("X 和 y 的样本数不一致")

        X_design = self._add_intercept(X_2d)
        n_features = X_design.shape[1]

        # 构造正则矩阵
        reg = self.alpha * np.eye(n_features)

        # 截距项通常不做正则
        if self.fit_intercept:
            reg[0, 0] = 0.0

        beta_hat = np.linalg.solve(X_design.T @ X_design + reg, X_design.T @ y_1d)

        if self.fit_intercept:
            self.intercept_ = float(beta_hat[0])
            self.coef_ = beta_hat[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta_hat

        self.is_fitted_ = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        """
        if not self.is_fitted_:
            raise RuntimeError("模型尚未拟合，请先调用 fit")

        if X.ndim == 2:
            return X @ self.coef_ + self.intercept_

        if X.ndim == 3:
            t_periods, n_stocks, _ = X.shape
            X_2d = X.reshape(t_periods * n_stocks, -1)
            y_pred = X_2d @ self.coef_ + self.intercept_
            return y_pred.reshape(t_periods, n_stocks)

        raise ValueError(f"X 仅支持二维或三维数组，当前维度为 {X.ndim}")

    def fit_predict(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        """
        拟合后直接预测
        """
        self.fit(X_train, y_train)
        return self.predict(X_test)


def run_linear_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    use_ridge: bool = True,
    ridge_alpha: float = 1.0,
    fit_intercept: bool = True,
) -> Dict[str, np.ndarray]:
    """
    统一运行线性 baseline

    返回
    ----
    result : dict
        {
            "model": 已训练模型,
            "y_pred": 测试集预测值
        }
    """
    if use_ridge:
        model = RidgeFactorModel(alpha=ridge_alpha, fit_intercept=fit_intercept)
    else:
        model = LinearFactorModel(fit_intercept=fit_intercept)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "model": model,
        "y_pred": y_pred,
    }