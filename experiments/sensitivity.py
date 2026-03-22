"""
敏感性分析模块

支持：
1. gamma 敏感性（邻居信号强度）
2. noise 敏感性（噪声水平）
3. k 敏感性（邻居数）

说明：
- 默认使用“图有效场景”
- 输出每组参数对应的模型表现表
"""

import copy
import numpy as np
import pandas as pd

import config

from data.simulate_data import simulate_factor_panel
from data.generate_returns import generate_returns_with_graph_signal
from data.split_data import time_series_split

from graph.build_graph import build_dynamic_graphs_from_factors

from features.preprocess import preprocess_panel
from features.raw_features import align_features_and_target
from features.graph_features import build_panel_graph_features

from models.linear_model import run_linear_baseline

from evaluation.metrics import evaluate_panel_predictions
from evaluation.compare_models import compare_model_outputs

from utils.seed import set_seed
from utils.logger import get_logger, log_section


def _run_single_linear_comparison(
    gamma: float,
    noise_std: float,
    k_neighbors: int,
):
    """
    运行单组参数下的线性模型对比：
    raw_linear vs graph_linear
    """
    set_seed(config.SEED)

    # 1. 数据
    data_dict = simulate_factor_panel(
        n_stocks=config.N_STOCKS,
        t_periods=config.T_PERIODS,
        k_features=config.K_FEATURES,
        n_groups=config.N_GROUPS,
        rho_group=config.RHO_GROUP,
        sigma_group=config.SIGMA_GROUP,
        sigma_idio=config.SIGMA_IDIO,
        seed=config.SEED,
    )
    X = preprocess_panel(data_dict["X"])

    # 2. 图
    A = build_dynamic_graphs_from_factors(
        X=X,
        graph_type=config.GRAPH_TYPE,
        k=k_neighbors,
        tau=config.TAU,
        symmetrize=True,
        add_self_loop=config.ADD_SELF_LOOP,
        row_norm=not config.NORMALIZE_ADJ,
        gcn_norm=config.NORMALIZE_ADJ,
    )

    # 3. 收益
    y = generate_returns_with_graph_signal(
        X=X,
        A_true=A,
        beta_self=config.BETA_SELF,
        beta_neighbor=config.BETA_NEIGHBOR,
        gamma=gamma,
        noise_std=noise_std,
        seed=config.SEED,
    )

    # 4. 对齐
    aligned = align_features_and_target(X, y, lag=1)
    X_aligned = aligned["X_aligned"]
    y_aligned = aligned["y_aligned"]
    A_aligned = A[:-1]

    X_graph = build_panel_graph_features(
        X=X_aligned,
        A=A_aligned,
        num_hops=config.NUM_HOPS,
        include_self=config.CONCAT_FEATURES,
    )

    # 5. 划分
    split_raw = time_series_split(
        X=X_aligned,
        y=y_aligned,
        train_ratio=config.TRAIN_RATIO,
        valid_ratio=config.VALID_RATIO,
        test_ratio=config.TEST_RATIO,
    )
    split_graph = time_series_split(
        X=X_graph,
        y=y_aligned,
        train_ratio=config.TRAIN_RATIO,
        valid_ratio=config.VALID_RATIO,
        test_ratio=config.TEST_RATIO,
    )

    # 6. raw
    raw_res = run_linear_baseline(
        X_train=split_raw["X_train"],
        y_train=split_raw["y_train"],
        X_test=split_raw["X_test"],
        use_ridge=config.USE_RIDGE,
        ridge_alpha=config.RIDGE_ALPHA,
    )
    raw_metrics = evaluate_panel_predictions(split_raw["y_test"], raw_res["y_pred"])

    # 7. graph
    graph_res = run_linear_baseline(
        X_train=split_graph["X_train"],
        y_train=split_graph["y_train"],
        X_test=split_graph["X_test"],
        use_ridge=config.USE_RIDGE,
        ridge_alpha=config.RIDGE_ALPHA,
    )
    graph_metrics = evaluate_panel_predictions(split_graph["y_test"], graph_res["y_pred"])

    return {
        "raw_linear": raw_metrics,
        "graph_linear": graph_metrics,
    }


def run_gamma_sensitivity(gamma_list):
    """
    邻居信号强度 gamma 敏感性分析

    返回
    ----
    df : DataFrame
        每个 gamma 下 raw / graph 模型表现
    """
    logger = get_logger(name="sensitivity_gamma")
    log_section(logger, "敏感性分析：gamma")

    rows = []
    for gamma in gamma_list:
        logger.info(f"运行 gamma={gamma:.4f}")
        out = _run_single_linear_comparison(
            gamma=gamma,
            noise_std=config.NOISE_STD,
            k_neighbors=config.K_NEIGHBORS,
        )

        rows.append({
            "gamma": gamma,
            "raw_r2": out["raw_linear"]["r2"],
            "graph_r2": out["graph_linear"]["r2"],
            "raw_mean_ic": out["raw_linear"]["mean_ic"],
            "graph_mean_ic": out["graph_linear"]["mean_ic"],
            "delta_r2": out["graph_linear"]["r2"] - out["raw_linear"]["r2"],
            "delta_ic": out["graph_linear"]["mean_ic"] - out["raw_linear"]["mean_ic"],
        })

    return pd.DataFrame(rows)


def run_noise_sensitivity(noise_list):
    """
    噪声水平敏感性分析
    """
    logger = get_logger(name="sensitivity_noise")
    log_section(logger, "敏感性分析：噪声水平")

    rows = []
    for noise_std in noise_list:
        logger.info(f"运行 noise_std={noise_std:.4f}")
        out = _run_single_linear_comparison(
            gamma=config.GAMMA,
            noise_std=noise_std,
            k_neighbors=config.K_NEIGHBORS,
        )

        rows.append({
            "noise_std": noise_std,
            "raw_r2": out["raw_linear"]["r2"],
            "graph_r2": out["graph_linear"]["r2"],
            "raw_mean_ic": out["raw_linear"]["mean_ic"],
            "graph_mean_ic": out["graph_linear"]["mean_ic"],
            "delta_r2": out["graph_linear"]["r2"] - out["raw_linear"]["r2"],
            "delta_ic": out["graph_linear"]["mean_ic"] - out["raw_linear"]["mean_ic"],
        })

    return pd.DataFrame(rows)


def run_k_sensitivity(k_list):
    """
    邻居数量 k 敏感性分析
    """
    logger = get_logger(name="sensitivity_k")
    log_section(logger, "敏感性分析：邻居数量 k")

    rows = []
    for k_neighbors in k_list:
        logger.info(f"运行 k={k_neighbors}")
        out = _run_single_linear_comparison(
            gamma=config.GAMMA,
            noise_std=config.NOISE_STD,
            k_neighbors=k_neighbors,
        )

        rows.append({
            "k_neighbors": k_neighbors,
            "raw_r2": out["raw_linear"]["r2"],
            "graph_r2": out["graph_linear"]["r2"],
            "raw_mean_ic": out["raw_linear"]["mean_ic"],
            "graph_mean_ic": out["graph_linear"]["mean_ic"],
            "delta_r2": out["graph_linear"]["r2"] - out["raw_linear"]["r2"],
            "delta_ic": out["graph_linear"]["mean_ic"] - out["raw_linear"]["mean_ic"],
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    gamma_df = run_gamma_sensitivity([0.0, 0.2, 0.5, 0.8, 1.0])
    print("\nGamma 敏感性结果：")
    print(gamma_df)

    noise_df = run_noise_sensitivity([0.5, 1.0, 1.5, 2.0])
    print("\nNoise 敏感性结果：")
    print(noise_df)

    k_df = run_k_sensitivity([3, 5, 10, 15, 20])
    print("\nk 敏感性结果：")
    print(k_df)