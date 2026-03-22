"""
GNN 模型模块

当前提供：
1. GCNLayer：单层图卷积
2. GCNRegressor：用于收益预测的 GCN 回归模型
3. GraphFeatureExtractor：仅提取图增强因子，不直接输出收益

说明：
- 输入单期图数据：
    X_t: (N, K)
    A_t: (N, N)
- 输出：
    pred: (N,)
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """
    单层图卷积

    数学形式：
        H = A_norm X W + b
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, X: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        X : Tensor, shape = (N, in_dim)
            节点特征矩阵
        A_norm : Tensor, shape = (N, N)
            归一化邻接矩阵

        返回
        ----
        H : Tensor, shape = (N, out_dim)
            更新后的节点表示
        """
        AX = torch.matmul(A_norm, X)
        H = self.linear(AX)
        return H


class GCNRegressor(nn.Module):
    """
    多层 GCN 回归模型

    结构：
        多层 H = ReLU( GCN(X, A) )
        层间 Dropout
        最后一层 GCN 直接输出预测值

    输出为每个节点一个实数预测值
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int = 1,
        num_layers: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        # 第一层
        self.layers.append(GCNLayer(in_dim=in_dim, out_dim=hidden_dim))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(in_dim=hidden_dim, out_dim=hidden_dim))
            
        # 最后一层
        self.layers.append(GCNLayer(in_dim=hidden_dim, out_dim=out_dim))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        X : Tensor, shape = (N, in_dim)
        A_norm : Tensor, shape = (N, N)

        返回
        ----
        pred : Tensor, shape = (N,)
        """
        H = X
        # 遍历前 num_layers - 1 层（应用 ReLU 和 Dropout）
        for i in range(self.num_layers - 1):
            H = self.layers[i](H, A_norm)
            H = F.relu(H)
            H = self.dropout(H)

        # 最后一层直接输出
        out = self.layers[-1](H, A_norm)
        pred = out.squeeze(-1)
        return pred


class GraphFeatureExtractor(nn.Module):
    """
    图特征提取器

    用法：
    - 将 GNN 作为“因子增强器”而不是直接预测器
    - 输出节点增强表示 Z，可与原始因子拼接后输入线性模型

    结构：
        Z = ReLU( GCN(X, A) )
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.gcn = GCNLayer(in_dim=in_dim, out_dim=hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        X : Tensor, shape = (N, in_dim)
        A_norm : Tensor, shape = (N, N)

        返回
        ----
        Z : Tensor, shape = (N, hidden_dim)
            图增强后的节点表示
        """
        Z = self.gcn(X, A_norm)
        Z = F.relu(Z)
        Z = self.dropout(Z)
        return Z


class GATLayer(nn.Module):
    """
    占位版 GAT 层接口

    说明：
    - 当前项目主线先用 GCN 即可
    - 若你后面需要，我可以再给你补完整的 GAT 实现
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("当前版本未实现 GATLayer，主实验请先使用 GCNRegressor。")


class GATRegressor(nn.Module):
    """
    占位版 GAT 回归器
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("当前版本未实现 GATRegressor，主实验请先使用 GCNRegressor。")