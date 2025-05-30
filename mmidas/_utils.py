import os
import random
import time
import warnings
from typing import List

import numpy as np
import torch as th

def set_seeds(s: int) -> None:
    if th.cuda.is_available():
        th.cuda.manual_seed(s)
    th.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)

def params_is_equal(ps1: dict[str, th.Tensor], ps2: dict[str, th.Tensor]) -> bool:
    return set(ps1.keys()) == set(ps2.keys()) and all(th.equal(ps1[k], ps2[k]) for k in ps1)
