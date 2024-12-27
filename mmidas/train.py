
import time

from torch.nn import functional as F
import tqdm

# TODO: Refactor with tensordict
def generic_train(model, opt, data, epochs=1, loss_fn=F.nll_loss, device=None):
    if device is None:
        device = next(model.parameters()).device

    model = model.to(device)

    total_loss = 0
    n_samples = 0
    tic = time.time()

    model.train()
    # for _ in range(epochs):
    for epoch in range(epochs):
        for (x, y) in (pbar := tqdm.tqdm(data, leave=False, desc=f'Epoch {epoch + 1}/{epochs}')):
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            opt.step()

            batch_loss = loss.item()
            batch_acc = (y_hat.argmax(dim=-1) == y).float().mean().item()

            # pbar.set_description(f'train/loss: {loss.item():.4f} | train/acc: {batch_acc:.4f}')
            pbar.set_description(f'Epoch {epoch + 1}/{epochs} | loss: {batch_loss:.4f} | acc: {batch_acc:.4f}')

            total_loss += loss.item()
            n_samples += len(x)

    print('train time:', time.time() - tic)
    print('final loss:', batch_loss)
    print('final acc:', batch_acc)

    return total_loss / n_samples