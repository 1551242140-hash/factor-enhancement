"""
实验C：错图 / 误导场景

场景设定：
- 收益由自身因子 + 隐藏信号决定
- 但构图基于显式因子相似性，和真实收益传播结构错配
- 预期：图增强方法可能无效，甚至比 baseline 更差
"""

import config

from data.simulate_data import simulate_factor_panel, simulate_hidden_signal_from_groups
from data.generate_returns import generate_returns_misaligned_graph
from data.split_data import time_series_split

from graph.build_graph import build_dynamic_graphs_from_factors

from features.preprocess import preprocess_panel
from features.raw_features import align_features_and_target
from features.graph_features import build_panel_graph_features

from models.linear_model import run_linear_baseline
from models.gnn_model import GCNRegressor
from models.trainer import train_gnn, predict_gnn

from evaluation.metrics import evaluate_panel_predictions
from evaluation.portfolio import evaluate_portfolio_performance
from evaluation.compare_models import compare_model_outputs, print_model_comparison

from utils.seed import set_seed
from utils.logger import get_logger, log_section


def run_experiment_scenario_c(delta_hidden: float = 1.0):
    """
    运行错图场景实验

    参数
    ----
    delta_hidden : float
        隐藏信号强度
    """
    set_seed(config.SEED)
    logger = get_logger(name="exp_scenario_c")

    log_section(logger, "实验C：错图/误导场景（图结构与真实收益机制错配）")

    # 1. 生成因子
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
    group_ids = data_dict["group_ids"]

    # 2. 构图（基于显式因子）
    A = build_dynamic_graphs_from_factors(
        X=X,
        graph_type=config.GRAPH_TYPE,
        k=config.K_NEIGHBORS,
        tau=config.TAU,
        symmetrize=True,
        add_self_loop=config.ADD_SELF_LOOP,
        row_norm=not config.NORMALIZE_ADJ,
        gcn_norm=config.NORMALIZE_ADJ,
    )

    # 3. 生成隐藏信号
    hidden_signal = simulate_hidden_signal_from_groups(
        group_ids=group_ids,
        t_periods=config.T_PERIODS,
        n_groups=config.N_GROUPS,
        rho_hidden=0.8,
        sigma_hidden=1.0,
        sigma_idio=0.3,
        seed=config.SEED + 123,
    )

    # 4. 收益由自身因子 + 隐藏信号决定
    y = generate_returns_misaligned_graph(
        X=X,
        hidden_signal=hidden_signal,
        beta_self=config.BETA_SELF,
        delta_hidden=delta_hidden,
        noise_std=config.NOISE_STD,
        seed=config.SEED,
    )

    # 5. 对齐
    aligned_raw = align_features_and_target(X, y, lag=1)
    X_aligned = aligned_raw["X_aligned"]
    y_aligned = aligned_raw["y_aligned"]
    A_aligned = A[:-1]

    X_graph = build_panel_graph_features(
        X=X_aligned,
        A=A_aligned,
        num_hops=config.NUM_HOPS,
        include_self=config.CONCAT_FEATURES,
    )

    # 6. 切分
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

    split_gnn = time_series_split(
        X=X_aligned,
        y=y_aligned,
        A=A_aligned,
        train_ratio=config.TRAIN_RATIO,
        valid_ratio=config.VALID_RATIO,
        test_ratio=config.TEST_RATIO,
    )

    results = {}

    # 7. baseline
    logger.info("训练原始线性模型...")
    raw_res = run_linear_baseline(
        X_train=split_raw["X_train"],
        y_train=split_raw["y_train"],
        X_test=split_raw["X_test"],
        use_ridge=config.USE_RIDGE,
        ridge_alpha=config.RIDGE_ALPHA,
    )
    y_pred_raw = raw_res["y_pred"]

    results["raw_linear"] = {
        "metrics": evaluate_panel_predictions(split_raw["y_test"], y_pred_raw),
        "portfolio": evaluate_portfolio_performance(
            y_true=split_raw["y_test"],
            y_pred=y_pred_raw,
            n_bins=config.N_QUANTILES,
            top_quantile=config.TOP_QUANTILE,
            bottom_quantile=config.BOTTOM_QUANTILE,
        ),
    }

    # 8. 图增强线性
    logger.info("训练图增强线性模型...")
    graph_res = run_linear_baseline(
        X_train=split_graph["X_train"],
        y_train=split_graph["y_train"],
        X_test=split_graph["X_test"],
        use_ridge=config.USE_RIDGE,
        ridge_alpha=config.RIDGE_ALPHA,
    )
    y_pred_graph = graph_res["y_pred"]

    results["graph_linear"] = {
        "metrics": evaluate_panel_predictions(split_graph["y_test"], y_pred_graph),
        "portfolio": evaluate_portfolio_performance(
            y_true=split_graph["y_test"],
            y_pred=y_pred_graph,
            n_bins=config.N_QUANTILES,
            top_quantile=config.TOP_QUANTILE,
            bottom_quantile=config.BOTTOM_QUANTILE,
        ),
    }

    # 9. GNN
    if config.USE_GNN:
        logger.info("训练 GNN 模型...")
        model = GCNRegressor(
            in_dim=split_gnn["X_train"].shape[-1],
            hidden_dim=config.HIDDEN_DIM,
            out_dim=1,
            num_layers=getattr(config, "NUM_GNN_LAYERS", 32),
            dropout=config.DROPOUT,
        )

        train_out = train_gnn(
            model=model,
            X_train=split_gnn["X_train"],
            A_train=split_gnn["A_train"],
            y_train=split_gnn["y_train"],
            X_valid=split_gnn["X_valid"],
            A_valid=split_gnn["A_valid"],
            y_valid=split_gnn["y_valid"],
            lr=config.LR,
            epochs=config.EPOCHS,
            weight_decay=config.WEIGHT_DECAY,
            patience=10,
            print_every=config.PRINT_EVERY,
            verbose=config.VERBOSE,
        )

        y_pred_gnn = predict_gnn(
            model=train_out["model"],
            X=split_gnn["X_test"],
            A=split_gnn["A_test"],
        )

        results["gnn"] = {
            "metrics": evaluate_panel_predictions(split_gnn["y_test"], y_pred_gnn),
            "portfolio": evaluate_portfolio_performance(
                y_true=split_gnn["y_test"],
                y_pred=y_pred_gnn,
                n_bins=config.N_QUANTILES,
                top_quantile=config.TOP_QUANTILE,
                bottom_quantile=config.BOTTOM_QUANTILE,
            ),
        }

    df = compare_model_outputs(results)
    print_model_comparison(df)

    return {
        "results": results,
        "comparison": df,
        "data": {
            "X": X_aligned,
            "y": y_aligned,
            "A": A_aligned,
            "hidden_signal": hidden_signal[:-1],
        },
    }


if __name__ == "__main__":
    run_experiment_scenario_c()