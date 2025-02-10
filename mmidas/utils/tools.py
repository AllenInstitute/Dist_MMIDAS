import os
from copy import deepcopy
from functools import lru_cache, partial
from pathlib import Path, PosixPath
from pprint import pprint
from typing import Any, assert_never

import sklearn.preprocessing as skp
import numpy as np
import requests
import toml


ROOT = "distributed-vae"
DATA_PATH = "data"
PATHS = "paths"
MILLION = 1e6


def join2(x, y) -> str:
    if y.startswith("/"):
        return y
    elif x.endswith("/"):
        return x + y
    else:
        return x + "/" + y


join_data = partial(join2, DATA_PATH)


def index_of(xs, t):
    if isinstance(xs, np.ndarray):
        return np.where(xs == t)[0][0]
    else:
        for i, x in enumerate(xs):
            if x == t:
                return i
        return -1


def indices_of(xs, ts):
    return [index_of(xs, t) for t in ts]


def join_path(xs) -> str:
    def go(xs, acc) -> str:
        match xs:
            case []:
                return acc
            case [x, *xs]:
                return go(xs, join2(acc, x))

    match xs:
        case []:
            raise ValueError("Must provide at least one path")
        case [x, *xs]:
            y = go(xs, x)
            assert y == os.path.join(*[x, *xs]), (y, os.path.join(*[x, *xs]))
            return y


# TODO: how to implement stat?
def path_exists(path):
    try:
        os.stat(path)
    except:
        return False
    v = True
    assert v == os.path.exists(path), (v, os.path.exists(path), path)
    return v


# TODO
def load_toml():
    raise NotImplementedError


def get_paths(path: str, dataset: str) -> dict[str, str]:
    """Loads dictionary with path names and any other variables set through xxx.toml

    Args:
        verbose (bool, optional): print paths

    Returns:
        config: dict
    """
    with open(path) as f:
        config = toml.load(f)

    config["paths"]["main_dir"] = ROOT

    # for k in [PATHS, dataset]:
    #     for l in config[k]:
    #         if path_exists(config[k][l]):
    #             config[k][l] = Path(config[k][l])

    return config


# L1 norm of each row
def l1(xss):
    return np.sum(np.abs(xss), axis=1)


def normalize(xss, f):
    return xss / f(xss)[:, None]


def normalize_cellxgene(xss) -> np.ndarray[Any, Any]:
    """Normalize based on number of input genes

    inpout args
        x (np.array): cell x gene matrix (cells along axis=0, genes along axis=1)

    return
        normalized gene expression matrix
    """
    return skp.normalize(xss, axis=1, norm="l1")


def logcpm2(xss):
    yss = np.log1p(normalize(xss, l1) * MILLION)
    assert np.allclose(yss, logcpm(xss)), (yss, logcpm(xss))
    return yss


def logcpm(x, scaler=1e6) -> np.ndarray[Any, Any]:
    """Log CPM normalization

    inpout args
        x (np.array): cell x gene matrix (cells along axis=0, genes along axis=1)
        scaler (float, optional): scaling factor for log CPM

    return
        normalized log CPM gene expression matrix
    """
    return np.log1p(normalize_cellxgene(x) * scaler)


def reorder_genes(x, chunksize=1000, eps=1e-1):
    t_gene = x.shape[1]
    print(t_gene)
    g_std, g_bin_std = [], []

    for i in range(int(t_gene // chunksize) + 1):
        ind0 = i * chunksize
        ind1 = np.min((t_gene, (i + 1) * chunksize))
        x_bin = np.where(x[:, ind0:ind1] > eps, 1, 0)
        g_std.append(np.std(x[:, ind0:ind1], axis=0))
        g_bin_std.append(np.std(x_bin, axis=0))

    g_std = np.concatenate(g_std)
    g_bin_std = np.concatenate(g_bin_std)
    g_ind = np.argsort(g_bin_std)
    g_ind = g_ind[np.sort(g_bin_std) > eps]
    print(len(g_ind))
    return g_ind[::-1]


def download_file(url, local_filename, chunk_size=10000):
    """Download a file from a URL and save it locally

    Args:
        url (str): URL of the file to download
        local_filename (str): Local path to save the file
        chunk_size (int, optional): Size of the chunks to download
    """
    # Send a HTTP GET request to the URL
    with requests.get(url, stream=True) as response:
        response.raise_for_status()  # Check if the request was successful
        # Open a local file in binary write mode
        with open(local_filename, "wb") as file:
            # Stream the content and write it in chunks to the local file
            for chunk in response.iter_content(chunk_size=chunk_size):
                file.write(chunk)
