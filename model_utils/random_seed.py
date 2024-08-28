import random

import torch
import numpy as np


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)  # 设置CPU生成随机数的种子，方便下次复现实验结果
    torch.cuda.manual_seed_all(seed)  # 为当前GPU设置随机种子
    np.random.seed(seed)  # 为numpy函数设置随机种子
    torch.backends.cudnn.deterministic = True  # 每次返回的卷积算法将是确定的，即默认算法。