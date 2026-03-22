"""
画图工具模块

作用：
1. 画训练损失曲线
2. 画 IC / RankIC 序列
3. 画多空组合累计收益
4. 画敏感性分析曲线

说明：
- 使用 matplotlib
- 不强制指定颜色，保持默认风格
"""

from typing import Dict, Optional
import os
import numpy as np
import matplotlib.pyplot as plt

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # 适用于 macOS
# plt.rcParams['font.sans-serif'] = ['SimHei'] # 适用于 Windows
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号


def _ensure_dir(save_path: Optional[str]) -> None:
    """
    如果提供保存路径，则确保目录存在
    """
    if save_path is None:
        return

    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)


def plot_loss_curve(
    train_losses,
    valid_losses=None,
    title: str = "训练损失曲线",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    绘制训练/验证损失曲线

    参数
    ----
    train_losses : array-like
        训练损失序列
    valid_losses : array-like or None
        验证损失序列
    """
    train_losses = np.asarray(train_losses).reshape(-1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")

    if valid_losses is not None and len(valid_losses) > 0:
        valid_losses = np.asarray(valid_losses).reshape(-1)
        plt.plot(valid_losses, label="Valid Loss")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)

    _ensure_dir(save_path)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()
    else:
        plt.close()


def plot_series(
    series,
    title: str = "时间序列图",
    xlabel: str = "Time",
    ylabel: str = "Value",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    绘制单条时间序列

    参数
    ----
    series : array-like
        序列数据
    """
    series = np.asarray(series).reshape(-1)

    plt.figure(figsize=(8, 5))
    plt.plot(series)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)

    _ensure_dir(save_path)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()
    else:
        plt.close()


def plot_cumulative_returns(
    cum_returns,
    title: str = "累计收益曲线",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Return",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    绘制累计收益曲线

    参数
    ----
    cum_returns : array-like
        累计收益序列
    """
    cum_returns = np.asarray(cum_returns).reshape(-1)

    plt.figure(figsize=(8, 5))
    plt.plot(cum_returns)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)

    _ensure_dir(save_path)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()
    else:
        plt.close()


def plot_group_returns(
    mean_group_returns,
    title: str = "分组平均收益",
    xlabel: str = "Group",
    ylabel: str = "Average Return",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    绘制分组平均收益柱状图

    参数
    ----
    mean_group_returns : array-like
        各组平均收益
    """
    mean_group_returns = np.asarray(mean_group_returns).reshape(-1)
    x = np.arange(1, len(mean_group_returns) + 1)

    plt.figure(figsize=(8, 5))
    plt.bar(x, mean_group_returns)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x)
    plt.grid(True, axis="y", alpha=0.3)

    _ensure_dir(save_path)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()
    else:
        plt.close()


def plot_model_comparison(
    comparison_dict: Dict[str, float],
    title: str = "模型指标对比",
    xlabel: str = "Model",
    ylabel: str = "Metric Value",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    绘制模型指标对比柱状图

    参数
    ----
    comparison_dict : dict
        例如：
        {
            "raw_linear": 0.03,
            "graph_linear": 0.05,
            "gnn": 0.06
        }
    """
    model_names = list(comparison_dict.keys())
    values = np.array(list(comparison_dict.values()), dtype=float)

    plt.figure(figsize=(8, 5))
    plt.bar(model_names, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)

    _ensure_dir(save_path)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()
    else:
        plt.close()


def plot_sensitivity_curve(
    x_values,
    y_values,
    title: str = "敏感性分析",
    xlabel: str = "Parameter",
    ylabel: str = "Performance",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    绘制敏感性分析曲线

    参数
    ----
    x_values : array-like
        参数取值
    y_values : array-like
        对应表现
    """
    x_values = np.asarray(x_values).reshape(-1)
    y_values = np.asarray(y_values).reshape(-1)

    if x_values.shape[0] != y_values.shape[0]:
        raise ValueError("x_values 和 y_values 长度必须一致")

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)

    _ensure_dir(save_path)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()
    else:
        plt.close()