"""
图结构有效性诊断测试模块

运行三个不同场景（图有效、图无效、错图），
汇总我们在 graph/diagnostics.py 中实现的 4 种统计检验指标，
输出一张汇总表格。
"""

import numpy as np
import pandas as pd
import os

import sys
import os

# 将项目根目录添加到系统路径中，以解决导入 config 和其他模块的问题
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data.simulate_data import simulate_factor_panel
from data.generate_returns import generate_returns_with_graph_signal
from features.preprocess import preprocess_panel
from features.raw_features import align_features_and_target
from graph.build_graph import build_dynamic_graphs_from_factors
from graph.diagnostics import (
    panel_morans_i,
    panel_graph_dirichlet_energy,
    panel_neighbor_residual_correlation,
    incremental_neighbor_regression_test,
)
from utils.seed import set_seed
from utils.logger import get_logger, log_section


def get_diagnostics_for_scenario(scenario_name: str, beta_neighbor: np.ndarray, A_type: str = "factor_knn"):
    """
    运行特定场景下的诊断检验
    """
    set_seed(config.SEED)
    
    # 1. 数据生成
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

    # 2. 构建真实图（用于生成收益）
    A_true = build_dynamic_graphs_from_factors(
        X=X,
        graph_type="factor_knn",
        k=config.K_NEIGHBORS,
        tau=config.TAU,
        symmetrize=True,
        add_self_loop=False,
        row_norm=False,
    )

    # 3. 生成收益
    y = generate_returns_with_graph_signal(
        X=X,
        A_true=A_true,
        beta_self=config.BETA_SELF,
        beta_neighbor=beta_neighbor,
        gamma=config.GAMMA,
        noise_std=config.NOISE_STD,
        seed=config.SEED,
    )

    # 4. 构造我们测试用的估计图 A_est
    if A_type == "random":
        # 错图场景：生成随机图
        A_est = np.random.rand(*A_true.shape)
        # 稀疏化处理，让密度和 kNN 差不多
        threshold = np.percentile(A_est, 100 - (config.K_NEIGHBORS / config.N_STOCKS * 100), axis=2, keepdims=True)
        A_est = (A_est >= threshold).astype(float)
    else:
        A_est = A_true

    # 对齐数据
    aligned = align_features_and_target(X, y, lag=1)
    X_aligned = aligned["X_aligned"]
    y_aligned = aligned["y_aligned"]
    A_aligned = A_est[:-1]

    T, N, K = X_aligned.shape

    # ==========================
    # 开始执行各项检验
    # ==========================
    
    # 1. Moran's I
    moran_series = panel_morans_i(y_aligned, A_aligned)
    mean_moran = np.mean(moran_series)

    # 2. Dirichlet Energy
    energy_series = panel_graph_dirichlet_energy(y_aligned, A_aligned, normalized=True)
    mean_energy = np.mean(energy_series)

    # 3. 邻居残差相关性
    # 首先计算只用自身因子时的残差
    residuals = np.zeros_like(y_aligned)
    for t in range(T):
        X_base = np.hstack([np.ones((N, 1)), X_aligned[t]])
        beta, _, _, _ = np.linalg.lstsq(X_base, y_aligned[t], rcond=None)
        y_pred_base = X_base @ beta
        residuals[t] = y_aligned[t] - y_pred_base
        
    resid_corr_series = panel_neighbor_residual_correlation(residuals, A_aligned)
    mean_resid_corr = np.mean(resid_corr_series)

    # 4. 增量回归检验 Delta R2
    delta_r2_list = []
    for t in range(T):
        reg_res = incremental_neighbor_regression_test(X_aligned[t], y_aligned[t], A_aligned[t])
        delta_r2_list.append(reg_res["delta_r2"])
    mean_delta_r2 = np.mean(delta_r2_list)

    return {
        "Scenario": scenario_name,
        "Moran_I": mean_moran,
        "Dirichlet_Energy": mean_energy,
        "Residual_Corr": mean_resid_corr,
        "Delta_R2": mean_delta_r2
    }


def run_all_diagnostics():
    logger = get_logger("diagnostics")
    log_section(logger, "图结构有效性诊断检验汇总")
    
    results = []
    
    # 场景A：图有效（有邻居效应，且用对图）
    logger.info("正在运行检验：场景A (图有效)...")
    res_a = get_diagnostics_for_scenario(
        scenario_name="A (图有效)", 
        beta_neighbor=config.BETA_NEIGHBOR, 
        A_type="factor_knn"
    )
    results.append(res_a)
    
    # 场景B：图无效（无邻居效应，用对图）
    logger.info("正在运行检验：场景B (图无效)...")
    res_b = get_diagnostics_for_scenario(
        scenario_name="B (图无效)", 
        beta_neighbor=np.zeros_like(config.BETA_NEIGHBOR), 
        A_type="factor_knn"
    )
    results.append(res_b)
    
    # 场景C：错图（有邻居效应，但用错图）
    logger.info("正在运行检验：场景C (错图)...")
    res_c = get_diagnostics_for_scenario(
        scenario_name="C (错图)", 
        beta_neighbor=config.BETA_NEIGHBOR, 
        A_type="random"
    )
    results.append(res_c)
    
    # 汇总输出
    df = pd.DataFrame(results).set_index("Scenario")
    
    print("\n========== 统计检验结果汇总 ==========")
    print(df.round(4))
    print("======================================\n")
    
    print("【指标说明】")
    print("1. Moran_I (莫兰指数): 衡量收益在图上的聚集性。值越大，图结构越能捕捉收益同质性。")
    print("2. Dirichlet_Energy (图平滑性能): 衡量收益在图上的不平滑程度。值越小，越适合GCN。")
    print("3. Residual_Corr (残差相关性): 基础模型残差在邻居间的相关性。显著为正代表图能提供增量Alpha。")
    print("4. Delta_R2 (增量R2): 引入邻居特征后，线性回归R2的平均提升。")
    
    # 保存结果
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/diagnostics_summary.csv", encoding="utf-8-sig")
    logger.info("检验结果已保存到 outputs/diagnostics_summary.csv")
    
    return df


if __name__ == "__main__":
    run_all_diagnostics()