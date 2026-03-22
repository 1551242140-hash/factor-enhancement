"""
GNN 训练模块

功能：
1. 训练 GCNRegressor
2. 在验证集上做 early stopping
3. 支持对面板数据逐期训练与预测

输入约定：
- X_train: (T, N, K)
- A_train: (T, N, N)
- y_train: (T, N)

说明：
- 每个时点视作一个图样本
- 训练时逐期遍历
"""

from typing import Dict, Optional, List
import copy
import numpy as np
import torch
import torch.nn as nn


def _to_tensor(x: np.ndarray, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    """
    numpy 转 torch tensor
    """
    return torch.tensor(x, dtype=dtype, device=device)


def evaluate_gnn_loss(
    model: nn.Module,
    X_eval: np.ndarray,
    A_eval: np.ndarray,
    y_eval: np.ndarray,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """
    计算验证集或测试集平均损失
    """
    model.eval()
    losses: List[float] = []

    with torch.no_grad():
        for t in range(X_eval.shape[0]):
            X_t = _to_tensor(X_eval[t], device=device)
            A_t = _to_tensor(A_eval[t], device=device)
            y_t = _to_tensor(y_eval[t], device=device)

            pred_t = model(X_t, A_t)
            loss_t = loss_fn(pred_t, y_t)
            losses.append(float(loss_t.item()))

    return float(np.mean(losses))


def train_gnn(
    model: nn.Module,
    X_train: np.ndarray,
    A_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: Optional[np.ndarray],
    A_valid: Optional[np.ndarray],
    y_valid: Optional[np.ndarray],
    lr: float = 1e-3,
    epochs: int = 50,
    weight_decay: float = 1e-4,
    patience: int = 10,
    print_every: int = 10,
    verbose: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """
    训练 GNN 模型

    参数
    ----
    model : nn.Module
        GNN 模型
    X_train, A_train, y_train : ndarray
        训练集面板数据
    X_valid, A_valid, y_valid : ndarray 或 None
        验证集面板数据；若为 None，则不做 early stopping
    lr : float
        学习率
    epochs : int
        训练轮数
    weight_decay : float
        L2 正则
    patience : int
        early stopping 容忍轮数
    print_every : int
        打印频率
    verbose : bool
        是否打印日志
    device : torch.device 或 None
        训练设备

    返回
    ----
    result : dict
        {
            "model": 训练后的最佳模型,
            "train_losses": list,
            "valid_losses": list
        }
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
    if verbose:
        print(f"GNN 训练使用设备: {device}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    train_losses = []
    valid_losses = []

    best_state = copy.deepcopy(model.state_dict())
    best_valid_loss = np.inf
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for t in range(X_train.shape[0]):
            X_t = _to_tensor(X_train[t], device=device)
            A_t = _to_tensor(A_train[t], device=device)
            y_t = _to_tensor(y_train[t], device=device)

            optimizer.zero_grad()
            pred_t = model(X_t, A_t)
            loss_t = loss_fn(pred_t, y_t)
            loss_t.backward()
            optimizer.step()

            epoch_losses.append(float(loss_t.item()))

        train_loss = float(np.mean(epoch_losses))
        train_losses.append(train_loss)

        # 验证集评估
        if X_valid is not None and A_valid is not None and y_valid is not None:
            valid_loss = evaluate_gnn_loss(
                model=model,
                X_eval=X_valid,
                A_eval=A_valid,
                y_eval=y_valid,
                loss_fn=loss_fn,
                device=device,
            )
            valid_losses.append(valid_loss)

            # early stopping
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_state = copy.deepcopy(model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1

            if verbose and (epoch % print_every == 0 or epoch == 1):
                print(
                    f"[Epoch {epoch:03d}] "
                    f"train_loss={train_loss:.6f}, valid_loss={valid_loss:.6f}"
                )

            if bad_epochs >= patience:
                if verbose:
                    print(f"触发 early stopping：连续 {patience} 轮验证集未改善。")
                break
        else:
            # 无验证集时仅打印训练损失
            if verbose and (epoch % print_every == 0 or epoch == 1):
                print(f"[Epoch {epoch:03d}] train_loss={train_loss:.6f}")

            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    return {
        "model": model,
        "train_losses": train_losses,
        "valid_losses": valid_losses,
    }


def predict_gnn(
    model: nn.Module,
    X: np.ndarray,
    A: np.ndarray,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    使用训练好的 GNN 对面板数据逐期预测

    参数
    ----
    model : nn.Module
        已训练模型
    X : ndarray, shape = (T, N, K)
    A : ndarray, shape = (T, N, N)

    返回
    ----
    y_pred : ndarray, shape = (T, N)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    preds = []

    with torch.no_grad():
        for t in range(X.shape[0]):
            X_t = _to_tensor(X[t], device=device)
            A_t = _to_tensor(A[t], device=device)

            pred_t = model(X_t, A_t)
            preds.append(pred_t.detach().cpu().numpy())

    y_pred = np.stack(preds, axis=0)
    return y_pred


def extract_graph_features(
    feature_extractor: nn.Module,
    X: np.ndarray,
    A: np.ndarray,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    使用 GraphFeatureExtractor 提取图增强因子

    参数
    ----
    feature_extractor : nn.Module
        图特征提取器
    X : ndarray, shape = (T, N, K)
    A : ndarray, shape = (T, N, N)

    返回
    ----
    Z : ndarray, shape = (T, N, hidden_dim)
        图增强后的面板特征
    """
    if device is None:
        device = next(feature_extractor.parameters()).device

    feature_extractor.eval()
    features = []

    with torch.no_grad():
        for t in range(X.shape[0]):
            X_t = _to_tensor(X[t], device=device)
            A_t = _to_tensor(A[t], device=device)

            z_t = feature_extractor(X_t, A_t)
            features.append(z_t.detach().cpu().numpy())

    Z = np.stack(features, axis=0)
    return Z