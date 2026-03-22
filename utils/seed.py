"""
随机种子工具模块

作用：
1. 固定 numpy / random / torch 的随机性
2. 提高实验可复现性
"""

import os
import random
import numpy as np


def set_seed(seed: int = 42, deterministic_torch: bool = True) -> None:
    """
    设置全局随机种子

    参数
    ----
    seed : int
        随机种子
    deterministic_torch : bool
        是否尽量让 PyTorch 使用确定性算法
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # 如果环境未安装 torch，则跳过
        pass