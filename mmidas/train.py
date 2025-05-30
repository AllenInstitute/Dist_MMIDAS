import time
import numpy as np

# TODO: feat: add augmenter
# TODO: feat: add reference prior
def train_mmidas(model, solver, train_loader, test_loader, spec):
    print("using device:", spec['device'])
    print(f"1 epoch = {len(train_loader)} batches")

    step = 0

    is_augment = spec['is_augment']
    is_ref_prior = spec['is_ref_prior']
    n_epochs = spec['n_epochs']
    n_arms = spec['n_arms']
    temp = spec['temp']
    c_p = spec['c_prior']
    c_onehot = spec['c_onehot']
    device = spec['device']

    model.train()
    for ep in range(n_epochs):
        for i_batch, (xs, i_xs) in enumerate(train_loader):
            tic = time.time()
            xs = xs.to(device)
            i_xs = i_xs.to(int)

            xs_aug = []
            for a in range(n_arms):
                if is_augment:
                    raise NotImplementedError("Augmentation is not implemented yet")
                else:
                    xs_aug.append(xs)

            if is_ref_prior:
                raise NotImplementedError("Reference prior is not implemented yet")
            else:
                c_bin = 0.0
                c_prior = 0.0

            solver.zero_grad()
            xs_rec, p_x, r_x, xs_low, qc, s, c, mu, logvar, log_qc = model(x=xs_aug, temp=temp, prior_c=c_prior)
            loss, loss_rec, loss_joint, entropy, dist_c, d_qc, kl_cont, min_var_0, ll, = model.loss(xs_rec, p_x, r_x, xs_aug, mu, logvar, qc, c, c_bin)
            loss.backward()
            solver.step()

            step += 1
            n_samples = len(xs)
            dt = time.time() - tic
            print(f"step: {step} | loss: {loss.item():.2f} | loss-rec-0: {loss_rec[0].item():.2f} | loss-rec-1: {loss_rec[1].item():.2f} | throughput: {n_samples/dt:.2f}it/s | dt: {dt*1000:.2f}ms")