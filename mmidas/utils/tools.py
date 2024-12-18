import os
import toml
import requests
from pathlib import Path, PosixPath
from functools import lru_cache
from typing import Any
from pprint import pprint
from copy import deepcopy

import numpy as np
import scipy.io as sio
from sklearn.preprocessing import normalize


def get_paths(config_filename: str, dataset: str = "files", verbose=False) -> dict[str, Any]:
    """Loads dictionary with path names and any other variables set through xxx.toml

    Args:
        verbose (bool, optional): print paths

    Returns:
        config: dict
    """
    cwd = Path(os.getcwd())

    with open(cwd / config_filename, "r") as f:
        config = toml.load(f)

    config["paths"]["main_dir"] = cwd

    for k in {"paths", dataset}:
        if k == dataset:
            print(f"loading {dataset} paths")
        for l in config[k]:
            if Path(config[k][l]).exists():
                config[k][l] = Path(config[k][l])

    if verbose:
        print("config:")
        pprint(config)

    return config


def normalize_cellxgene(x) -> np.ndarray[Any, Any]:
    """Normalize based on number of input genes

    inpout args
        x (np.array): cell x gene matrix (cells along axis=0, genes along axis=1)

    return
        normalized gene expression matrix
    """
    return normalize(x, axis=1, norm="l1")


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
