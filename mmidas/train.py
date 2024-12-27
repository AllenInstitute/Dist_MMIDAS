
from torch.nn import functional as F

def generic_train(model, opt, data, loss_fn=F.nll_loss, device=None):
    if device is None:
        device = next(model.parameters()).device

    model = model.to(device)

    total_loss = 0
    n_samples = 0

    model.train()
    for (x, y) in data:
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        loss.backward()
        opt.step()

        total_loss += loss.item()
        n_samples += len(x)


    return total_loss / n_samples