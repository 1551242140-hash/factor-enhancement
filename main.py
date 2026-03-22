"""
主程序入口

支持功能：
1. 运行单个实验场景
   - A：图有效
   - B：图无效
   - C：错图/误导
2. 运行敏感性分析
   - gamma
   - noise
   - k

使用方式：
--------
1. 直接运行默认场景：
    python main.py

2. 修改下方 RUN_MODE 和 EXPERIMENT_NAME

说明：
--------
本文件负责串联整个项目，不直接实现具体模型与数据逻辑。
"""

import os
import pandas as pd
import numpy as np

from experiments.exp_scenario_a import run_experiment_scenario_a
from experiments.exp_scenario_b import run_experiment_scenario_b
from experiments.exp_scenario_c import run_experiment_scenario_c
from experiments.sensitivity import (
    run_gamma_sensitivity,
    run_noise_sensitivity,
    run_k_sensitivity,
)

from utils.logger import get_logger, log_section


# =========================
# 运行模式设置
# =========================
# 可选：
# "experiment_all" -> 一次性跑完 A/B/C 三个场景并作对比
# "experiment"   -> 跑单个实验场景
# "sensitivity"  -> 跑敏感性分析
RUN_MODE = "sensitivity"

# 当 RUN_MODE = "experiment" 时可选：
# "A" -> 图有效
# "B" -> 图无效
# "C" -> 错图/误导
EXPERIMENT_NAME = "A"

# 当 RUN_MODE = "sensitivity" 时可选：
# "gamma" / "noise" / "k" / "all"
SENSITIVITY_NAME = "all"

# 是否保存结果
SAVE_RESULTS = True
SAVE_DIR = "outputs"


def ensure_dir(path: str) -> None:
    """
    若目录不存在则创建
    """
    os.makedirs(path, exist_ok=True)


def save_dataframe(df: pd.DataFrame, filename: str, save_dir: str = SAVE_DIR) -> str:
    """
    保存 DataFrame 为 CSV

    参数
    ----
    df : pd.DataFrame
        待保存结果表
    filename : str
        文件名
    save_dir : str
        保存目录

    返回
    ----
    save_path : str
        实际保存路径
    """
    ensure_dir(save_dir)
    save_path = os.path.join(save_dir, filename)
    df.to_csv(save_path, encoding="utf-8-sig", index=True)
    return save_path


def run_single_experiment(experiment_name: str):
    """
    运行单个实验场景

    参数
    ----
    experiment_name : str
        "A" / "B" / "C"

    返回
    ----
    result : dict
        实验返回结果
    """
    experiment_name = experiment_name.upper()

    if experiment_name == "A":
        return run_experiment_scenario_a()

    if experiment_name == "B":
        return run_experiment_scenario_b()

    if experiment_name == "C":
        return run_experiment_scenario_c()

    raise ValueError("EXPERIMENT_NAME 仅支持 'A'、'B'、'C'")


def run_single_sensitivity(sensitivity_name: str) -> pd.DataFrame:
    """
    运行单个敏感性分析

    参数
    ----
    sensitivity_name : str
        "gamma" / "noise" / "k"

    返回
    ----
    df : pd.DataFrame
        敏感性分析结果表
    """
    sensitivity_name = sensitivity_name.lower()

    if sensitivity_name == "gamma":
        return run_gamma_sensitivity([0.0, 0.2, 0.5, 0.8, 1.0])

    if sensitivity_name == "noise":
        return run_noise_sensitivity([0.5, 1.0, 1.5, 2.0])

    if sensitivity_name == "k":
        return run_k_sensitivity([3, 5, 10, 15, 20])

    raise ValueError("SENSITIVITY_NAME 仅支持 'gamma'、'noise'、'k'")


def main():
    """
    主函数
    """
    logger = get_logger(name="main")

    log_section(logger, "股票相似图是否增强因子信号：主程序开始")

    if RUN_MODE == "experiment":
        logger.info(f"当前运行模式：单场景实验 | 场景 = {EXPERIMENT_NAME}")
        result = run_single_experiment(EXPERIMENT_NAME)

        comparison_df = result["comparison"]
        logger.info("实验运行完成。")

        if SAVE_RESULTS:
            filename = f"comparison_experiment_{EXPERIMENT_NAME}.csv"
            save_path = save_dataframe(comparison_df, filename=filename)
            logger.info(f"结果已保存到：{save_path}")
            
            # 绘制模型 R2 和 IC 对比图
            from utils.plotting import plot_model_comparison
            
            r2_dict = comparison_df["r2"].to_dict()
            ic_dict = comparison_df["mean_ic"].to_dict()
            
            plot_model_comparison(
                r2_dict, 
                title=f"实验 {EXPERIMENT_NAME}：模型 R2 对比", 
                ylabel="R2",
                save_path=os.path.join(SAVE_DIR, f"experiment_{EXPERIMENT_NAME}_r2.png"),
                show=False
            )
            plot_model_comparison(
                ic_dict, 
                title=f"实验 {EXPERIMENT_NAME}：模型 IC 对比", 
                ylabel="Mean IC",
                save_path=os.path.join(SAVE_DIR, f"experiment_{EXPERIMENT_NAME}_ic.png"),
                show=False
            )
            logger.info("可视化图片已保存到 outputs/ 目录下。")

    elif RUN_MODE == "experiment_all":
        logger.info("当前运行模式：一次性运行 A/B/C 三个场景并作对比")
        
        results_all = {}
        for scenario in ["A", "B", "C"]:
            logger.info(f"\n===== 开始运行场景 {scenario} =====")
            res = run_single_experiment(scenario)
            
            # 从 comparison_df 中提取 graph_linear 的 R2 和 IC
            df = res["comparison"]
            results_all[f"Scenario_{scenario}"] = {
                "raw_linear_r2": df.loc["raw_linear", "r2"],
                "graph_linear_r2": df.loc["graph_linear", "r2"],
                "gnn_r2": df.loc["gnn", "r2"],
                "raw_linear_ic": df.loc["raw_linear", "mean_ic"],
                "graph_linear_ic": df.loc["graph_linear", "mean_ic"],
                "gnn_ic": df.loc["gnn", "mean_ic"],
            }
            
        summary_df = pd.DataFrame(results_all).T
        print("\n===== 场景 A/B/C 综合对比 =====")
        print(summary_df)
        print("===============================\n")
        
        if SAVE_RESULTS:
            save_path = save_dataframe(summary_df, filename="comparison_all_scenarios.csv")
            logger.info(f"汇总结果已保存到：{save_path}")
            
            # 为 A/B/C 的 R2 画一个对比柱状图
            import matplotlib.pyplot as plt
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
            
            scenarios = summary_df.index
            x = np.arange(len(scenarios))
            width = 0.25
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # R2 比较
            ax = axes[0]
            ax.bar(x - width, summary_df["raw_linear_r2"], width, label='Raw Linear')
            ax.bar(x, summary_df["graph_linear_r2"], width, label='Graph Linear')
            ax.bar(x + width, summary_df["gnn_r2"], width, label='GNN')
            ax.set_ylabel('R2 Score')
            ax.set_title('场景 A/B/C 性能对比 (R2)')
            ax.set_xticks(x)
            ax.set_xticklabels(scenarios)
            ax.legend()
            
            # IC 比较
            ax = axes[1]
            ax.bar(x - width, summary_df["raw_linear_ic"], width, label='Raw Linear')
            ax.bar(x, summary_df["graph_linear_ic"], width, label='Graph Linear')
            ax.bar(x + width, summary_df["gnn_ic"], width, label='GNN')
            ax.set_ylabel('Mean IC')
            ax.set_title('场景 A/B/C 性能对比 (Mean IC)')
            ax.set_xticks(x)
            ax.set_xticklabels(scenarios)
            ax.legend()
            
            plt.tight_layout()
            fig_path = os.path.join(SAVE_DIR, "comparison_all_scenarios.png")
            plt.savefig(fig_path, dpi=150)
            plt.close()
            logger.info("综合对比图已保存。")

    elif RUN_MODE == "sensitivity":
        if SENSITIVITY_NAME == "all":
            logger.info("当前运行模式：敏感性分析 | 类型 = all (gamma, noise, k)")
            sens_list = ["gamma", "noise", "k"]
        else:
            logger.info(f"当前运行模式：敏感性分析 | 类型 = {SENSITIVITY_NAME}")
            sens_list = [SENSITIVITY_NAME]

        for s_name in sens_list:
            logger.info(f"\n===== 开始敏感性分析: {s_name} =====")
            df = run_single_sensitivity(s_name)
    
            print(f"\n===== 敏感性分析结果：{s_name} =====")
            print(df.round(6))
            print("=====================================\n")
    
            if SAVE_RESULTS:
                filename = f"sensitivity_{s_name}.csv"
                save_path = save_dataframe(df, filename=filename)
                logger.info(f"结果已保存到：{save_path}")
                
                # 绘制敏感性分析曲线
                from utils.plotting import plot_sensitivity_curve
                
                # x 轴是第一列（比如 gamma, noise_std, k_neighbors）
                x_col = df.columns[0]
                
                # 画 R2 提升图
                plot_sensitivity_curve(
                    df[x_col], df["delta_r2"],
                    title=f"敏感性分析 ({s_name})：图增强 R2 提升",
                    xlabel=x_col,
                    ylabel="Delta R2",
                    save_path=os.path.join(SAVE_DIR, f"sensitivity_{s_name}_r2.png"),
                    show=False
                )
                
                # 画 IC 提升图
                plot_sensitivity_curve(
                    df[x_col], df["delta_ic"],
                    title=f"敏感性分析 ({s_name})：图增强 IC 提升",
                    xlabel=x_col,
                    ylabel="Delta IC",
                    save_path=os.path.join(SAVE_DIR, f"sensitivity_{s_name}_ic.png"),
                    show=False
                )
                logger.info(f"可视化图片已保存到 outputs/ 目录下 ({s_name})。")

    else:
        raise ValueError("RUN_MODE 仅支持 'experiment' 或 'sensitivity'")

    logger.info("主程序结束。")


if __name__ == "__main__":
    main()