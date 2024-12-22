import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from typing import Callable

_dataset_registry = {}

def register_dataset(name, data_fn):
    _dataset_registry[name] = data_fn

def load_data(dataset, *args, **kwargs):
    return _dataset_registry[dataset](*args, **kwargs)

def _generic_load(fn, transform, batch_size, test_batch_size=None):
    if test_batch_size is None:
        test_batch_size = batch_size

    train_data = fn('data', train=True, download=True, transform=transform)
    test_data = fn('data', train=False, download=True, transform=transform)

    train_data, val_data = _split_data(train_data)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def _split_data(train, split=0.8):
    train_size = int(split * len(train))
    return random_split(train, [train_size, len(train) - train_size])

def _load_mnist(batch_size,  test_batch_size=None):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,)),
                                    transforms.Lambda(lambda x: x.squeeze(0))])
    return _generic_load(datasets.MNIST, transform, batch_size, test_batch_size)
    

def _load_cifar10(batch_size, test_batch_size=None):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return _generic_load(datasets.CIFAR10, transform, batch_size, test_batch_size)


register_dataset('mnist', _load_mnist)
register_dataset('cifar10', _load_cifar10)
