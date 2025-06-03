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