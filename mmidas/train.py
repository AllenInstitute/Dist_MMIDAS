
import time

import torch as th
from torch.nn import functional as F
import tqdm

# TODO: Refactor with tensordict
def generic_train(model, opt, train_data, val_data, epochs, loss_fn, device, f):
    metrics = {
        'train_loss': 0,
        'train_acc': 0,
        'val_loss': None,
        'val_acc': None,
        'total_loss': 0,
        'n_samples': 0
    }
    tic = time.time()

    for epoch in range(epochs):
        model.train()

        desc = f'Epoch {epoch + 1}/{epochs}'
        desc += f' | loss: {metrics["train_loss"]:.4f} | acc: {metrics["train_acc"]:.4f}'
        if metrics['val_loss'] is not None:
            desc += f' | val_loss: {metrics["val_loss"]:.4f} | val_acc: {metrics["val_acc"]:.4f}'

        for x, y in (pbar := tqdm.tqdm(train_data, leave=False, desc=desc)):
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            y_hat = f(model)(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            opt.step()

            metrics['train_loss'] = loss.item()
            metrics['train_acc'] = th.argmax(y_hat, dim=-1).float().mean().item()
            metrics['total_loss'] += loss.item()
            metrics['n_samples'] += len(x)

            desc = f'Epoch {epoch + 1}/{epochs} | loss: {metrics["train_loss"]:.4f} | acc: {metrics["train_acc"]:.4f}'
            if metrics['val_loss'] is not None:
                desc += f' | val_loss: {metrics["val_loss"]:.4f} | val_acc: {metrics["val_acc"]:.4f}'
            pbar.set_description(desc)
        
        if val_data is not None:
            model.eval()
            with th.no_grad():
                metrics['val_loss'], metrics['val_acc'] = generic_eval(model, val_data, loss_fn=loss_fn, device=device, f=f)

    print(f'train completed in: {time.time() - tic:.2f}s')
    print(f'final train loss: {metrics["train_loss"]:.4f}')
    print(f'final train acc: {metrics["train_acc"]:.4f}')
    if val_data is not None:
        print(f'final val loss: {metrics["val_loss"]:.4f}')
        print(f'final val acc: {metrics["val_acc"]:.4f}')

    return metrics

def generic_eval(model, data, loss_fn=F.nll_loss, device=None, f=lambda x: x):
    if device is None:
        device = next(model.parameters()).device

    model = model.to(device)

    total_loss = 0
    n_correct = 0
    n_samples = 0

    with th.no_grad():
        model.eval()
        for (x, y) in data:
            x, y = x.to(device), y.to(device)

            y_hat = f(model)(x)
            loss = loss_fn(y_hat, y)

            n_correct += (y_hat.argmax(dim=-1) == y).float().sum().item()

            total_loss += loss.item()
            n_samples += len(x)

    return total_loss / n_samples, n_correct / n_samples