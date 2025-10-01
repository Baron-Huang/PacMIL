#### Author: Dr. Pan Huang
#### Email: mrhuangpan@163.com or pan.huang@polyu.edu.hk
#### Department: PolyU, HK
import torch
import numpy as np
import random

########################## seed_function #########################
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = True
