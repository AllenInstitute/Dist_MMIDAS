import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.core.random as random
import numpy as np

# %%

def expit(x):
    return 1 / (1 + np.exp(-np.array(x)))

# %%

def make_dspec(n_features, n_samples):
    return {
        'n_features': n_features,
        'n_samples': n_samples,
        'type': 'DatasetSpec'
    }

def dspec_n_features(dspec):
    return dspec['n_features']

def dspec_n_samples(dspec):
    return dspec['n_samples']

seed = 546
ds = make_dspec(n_features=10, n_samples=100)

np.random.seed(seed)

true_params = mx.array(np.random.randn(dspec_n_features(ds))) # from beta distribution
inputs = mx.array(np.random.randn(dspec_n_samples(ds), dspec_n_features(ds)))
targets = mx.array(np.random.rand(dspec_n_samples(ds)) < expit(np.dot(inputs, true_params)), dtype=mx.int32)
