import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


from typing import Callable

def visualize(dataloader, class_names, n=16, figsize=(10, 10)):
    """
    Visualize n random samples from a dataloader with their class labels.
    
    Args:
        dataloader: PyTorch DataLoader object
        class_names: Dictionary mapping class indices to class names
        n: Number of samples to visualize (default: 16)
        figsize: Figure size for the plot (default: (10, 10))
    """
    # Get a batch of images
    images, labels = next(iter(dataloader))
    
    # Generate random indices
    total_samples = len(images)
    indices = np.random.choice(total_samples, min(n, total_samples), replace=False)
    
    # Select random samples
    selected_images = images[indices]
    selected_labels = labels[indices]
    
    # Handle grayscale images
    if selected_images.shape[1] == 1:
        selected_images = selected_images.repeat(1, 3, 1, 1)
    
    # Create subplot for each image with its label
    rows = int(np.sqrt(n))
    cols = int(np.ceil(n / rows))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.ravel()
    
    for idx, (img, label) in enumerate(zip(selected_images, selected_labels)):
        # Convert from (C,H,W) to (H,W,C)
        if img.shape[0] == 3 or img.shape[0] == 1:
            img = img.permute(1, 2, 0)
        img = img.numpy()
        
        # If single channel, squeeze the channel dimension
        if img.shape[-1] == 1:
            img = img.squeeze()
            
        axes[idx].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        axes[idx].axis('off')
        axes[idx].set_title(f'Class: {class_names[label.item()]}')
    
    # Hide empty subplots
    for idx in range(len(selected_images), len(axes)):
        axes[idx].axis('off')
        
    plt.tight_layout()
    plt.show()

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
                                    transforms.Normalize((0.1307,), (0.3081,))])
    return _generic_load(datasets.MNIST, transform, batch_size, test_batch_size)
    

def _load_cifar10(batch_size, test_batch_size=None):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return _generic_load(datasets.CIFAR10, transform, batch_size, test_batch_size)


register_dataset('mnist', _load_mnist)
register_dataset('cifar10', _load_cifar10)
