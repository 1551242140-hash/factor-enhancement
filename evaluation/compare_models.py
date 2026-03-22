"""
模型结果对比模块

作用：
1. 汇总不同模型的预测评估结果
2. 汇总不同模型的组合表现
3. 输出统一对比表

说明：
- 输入通常是多个模型的结果字典
- 输出为便于打印和保存的对比字典 / DataFrame
"""

from typing import Dict, Any
import numpy as np
import pandas as pd


def summarize_single_model_result(
    metrics_result: Dict[str, Any],
    portfolio_result: Dict[str, Any],
) -> Dict[str, float]:
    """
    将单个模型的评估结果压缩为一行摘要

    参数
    ----
    metrics_result : dict
        来自 evaluation.metrics.evaluate_panel_predictions 的输出
    portfolio_result : dict
        来自 evaluation.portfolio.evaluate_portfolio_performance 的输出

    返回
    ----
    summary : dict
    """
    summary = {
        "mse": float(metrics_result.get("mse", np.nan)),
        "mae": float(metrics_result.get("mae", np.nan)),
        "r2": float(metrics_result.get("r2", np.nan)),
        "mean_ic": float(metrics_result.get("mean_ic", np.nan)),
        "ic_ir": float(metrics_result.get("ic_ir", np.nan)),
        "mean_rank_ic": float(metrics_result.get("mean_rank_ic", np.nan)),
        "rank_ic_ir": float(metrics_result.get("rank_ic_ir", np.nan)),
        "mean_ls_return": float(portfolio_result.get("mean_ls_return", np.nan)),
        "ls_sharpe": float(portfolio_result.get("ls_sharpe", np.nan)),
    }

    mean_group_returns = portfolio_result.get("mean_group_returns", None)
    if mean_group_returns is not None:
        mean_group_returns = np.asarray(mean_group_returns).reshape(-1)
        for i, val in enumerate(mean_group_returns):
            summary[f"group_{i + 1}_ret"] = float(val)

    return summary


def compare_model_outputs(results_dict: Dict[str, Dict[str, Dict[str, Any]]]) -> pd.DataFrame:
    """
    汇总多个模型的结果，输出为 DataFrame

    输入格式示例
    ----------
    results_dict = {
        "raw_linear": {
            "metrics": {...},
            "portfolio": {...}
        },
        "graph_linear": {
            "metrics": {...},
            "portfolio": {...}
        },
        "gnn": {
            "metrics": {...},
            "portfolio": {...}
        }
    }

    返回
    ----
    df : pd.DataFrame
        行为模型名，列为各项指标
    """
    rows = {}

    for model_name, result in results_dict.items():
        if "metrics" not in result:
            raise KeyError(f"{model_name} 缺少 'metrics' 字段")
        if "portfolio" not in result:
            raise KeyError(f"{model_name} 缺少 'portfolio' 字段")

        rows[model_name] = summarize_single_model_result(
            metrics_result=result["metrics"],
            portfolio_result=result["portfolio"],
        )

    df = pd.DataFrame.from_dict(rows, orient="index")
    return df


def print_model_comparison(df: pd.DataFrame, round_digits: int = 6) -> None:
    """
    打印模型对比表

    参数
    ----
    df : pd.DataFrame
        compare_model_outputs 输出结果
    round_digits : int
        小数位数
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df 必须是 pandas.DataFrame")

    print("\n========== 模型表现对比 ==========")
    print(df.round(round_digits))
    print("=================================\n")


def save_model_comparison(df: pd.DataFrame, path: str) -> None:
    """
    保存模型对比表到 CSV

    参数
    ----
    df : pd.DataFrame
    path : str
        文件路径
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df 必须是 pandas.DataFrame")

    df.to_csv(path, encoding="utf-8-sig")


def compare_two_models_delta(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
) -> pd.Series:
    """
    比较两个模型的指标差值：model_b - model_a

    参数
    ----
    df : pd.DataFrame
        模型对比表
    model_a : str
        基准模型名
    model_b : str
        对比模型名

    返回
    ----
    delta : pd.Series
        各指标差值
    """
    if model_a not in df.index:
        raise KeyError(f"{model_a} 不在 DataFrame 索引中")
    if model_b not in df.index:
        raise KeyError(f"{model_b} 不在 DataFrame 索引中")

    delta = df.loc[model_b] - df.loc[model_a]
    delta.name = f"{model_b} - {model_a}"
    return delta