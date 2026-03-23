# 股票相似图因子增强项目 (Stock Graph Factor)

本项目旨在探索和验证**股票相似图结构**（如基于因子的 kNN 图）是否能够有效增强传统的量化多因子模型。通过构建不同设定的模拟面板数据，我们对比了**原始线性模型**、**图增强线性模型**以及 **GNN (图神经网络) 模型** 在不同场景下的预测表现。

## 项目结构

```text
stock_graph_factor/
├── main.py                      # 主程序入口，负责串联实验与敏感性分析
├── config.py                    # 全局参数配置 (数据生成、模型超参等)
├── README.md                    # 项目说明文档
│
├── data/                        # 数据模块
│   ├── simulate_data.py         # 模拟多因子面板数据生成
│   ├── generate_returns.py      # 收益生成机制 (支持图有效、图无效等设定)
│   └── split_data.py            # 训练集 / 验证集 / 测试集 划分
│
├── graph/                       # 图构造模块
│   ├── build_graph.py           # 构图方法：基于因子距离构建 kNN 图
│   ├── graph_utils.py           # 图工具：邻接矩阵归一化等
│   └── diagnostics.py           # (待完善) 图结构诊断检验
│
├── features/                    # 特征工程模块
│   ├── raw_features.py          # 原始因子整理与对齐
│   ├── graph_features.py        # 基于图的特征聚合 (如邻居特征平均 AX)
│   └── preprocess.py            # 数据预处理 (标准化等)
│
├── models/                      # 模型模块
│   ├── linear_model.py          # OLS / Ridge 线性基准模型
│   ├── gnn_model.py             # GCN 回归模型
│   ├── ml_model.py              # (待完善) 传统机器学习模型
│   └── trainer.py               # GNN 模型训练与评估流程 (支持自动使用 GPU/MPS)
│
├── evaluation/                  # 评估模块
│   ├── metrics.py               # 评价指标 (MSE, MAE, R2, IC, RankIC)
│   ├── portfolio.py             # 投资组合表现 (多空收益、分组收益)
│   └── compare_models.py        # 模型对比与结果汇总
│
├── experiments/                 # 实验场景模块
│   ├── exp_scenario_a.py        # 实验 A：图有效场景 (邻居信息对收益有贡献)
│   ├── exp_scenario_b.py        # 实验 B：图无效场景 (收益仅由自身因子决定)
│   ├── exp_scenario_c.py        # 实验 C：错图/误导场景 (图结构与真实收益机制错配)
│   ├── sensitivity.py           # 敏感性分析 (gamma 信号强度, noise 噪声水平, k 邻居数)
│   └── run_diagnostics.py       # 图结构有效性事前诊断检验
│
├── outputs/                     # 实验结果与可视化图片输出目录 (自动生成)
│
└── utils/                       # 通用工具模块
    ├── logger.py                # 日志记录工具
    ├── plotting.py              # 结果可视化工具 (支持中文显示)
    └── seed.py                  # 全局随机种子设置，保证可复现性
```

## 核心功能

1. **多场景实验对比**：
   - **场景 A（图有效）**：股票的未来收益不仅由自身因子决定，还受到图中邻居股票的因子影响。此时图增强模型和 GNN 理论上表现更优。
   - **场景 B（图无效）**：股票收益完全由自身因子决定，图结构仅为噪音。用于测试复杂模型是否会过拟合。
   - **场景 C（错图/误导）**：使用的图结构与真实的收益溢出机制完全不一致。测试模型在面对错误图结构时的鲁棒性。

2. **敏感性分析**：
   - 探究邻居信号强度（`gamma`）、噪声水平（`noise`）以及构图时选择的邻居数量（`k`）对图增强效果的具体影响。

3. **可视化分析**：
   - 自动绘制不同模型间的 $R^2$、Mean IC 对比柱状图。
   - 自动绘制敏感性分析参数变化带来的增量效果折线图。

## 快速开始

### 1. 环境依赖
推荐使用 Python 3.8+ 环境，安装以下依赖：
```bash
pip install torch pandas numpy scikit-learn scipy networkx matplotlib
```
*注：代码已适配 Mac 的 MPS 加速以及 Windows/Linux 的 CUDA 加速。*

### 2. 运行单场景实验
在 `main.py` 中将 `RUN_MODE` 设置为 `"experiment"`，并指定 `EXPERIMENT_NAME` 为 `"A"`、`"B"` 或 `"C"`：

```python
# main.py
RUN_MODE = "experiment"
EXPERIMENT_NAME = "A"  # 运行图有效场景
SAVE_RESULTS = True    # 开启结果保存与可视化图片生成
```
执行：
```bash
python main.py
```
运行完成后，可在终端查看模型表现对比表格，并在 `outputs/` 目录下找到生成的 CSV 报表与对比图表（如 `experiment_A_r2.png`）。

### 3. 运行敏感性分析
在 `main.py` 中将 `RUN_MODE` 设置为 `"sensitivity"`，并指定需要分析的参数 `SENSITIVITY_NAME`（可选：`"gamma"`, `"noise"`, `"k"`）：

```python
# main.py
RUN_MODE = "sensitivity"
SENSITIVITY_NAME = "gamma"  # 测试邻居信号强度的敏感性
SAVE_RESULTS = True         # 开启结果保存与可视化图片生成
```
执行：
```bash
python main.py
```
程序会遍历不同的参数设定，并自动在 `outputs/` 目录下生成随参数变化的增益曲线图（如 `sensitivity_gamma_r2.png`）。

## 参数调整
你可以直接修改 `config.py` 来调整实验设定。核心参数包括：
- `N_STOCKS`, `T_PERIODS`: 股票数量与时间长度。
- `K_NEIGHBORS`: kNN 图的邻居数量 $K$。
- `GAMMA`: 邻居因子对收益的贡献权重。
- `NUM_GNN_LAYERS`: GNN 模型的网络层数（建议保持在 2~3 层以避免过平滑）。
- `LR`, `EPOCHS`: 神经网络的学习率与训练轮数。