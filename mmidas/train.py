import time
import wandb
import numpy as np

# TODO: feat: add augmenter
# TODO: feat: add reference prior
# TODO: feat: automatic logging of multiple rec losses
def train_mmidas(model, solver, train_loader, test_loader, spec):
    print("using device:", spec['device'])
    print(f"1 epoch = {len(train_loader)} batches")

    is_augment = spec['is_augment']
    is_ref_prior = spec['is_ref_prior']
    n_epochs = spec['n_epochs']
    n_arms = spec['n_arms']
    temp = spec['temp']
    c_p = spec['c_prior']
    c_onehot = spec['c_onehot']
    print_every = spec['print_every']
    device = spec['device']

    def train_step(model, solver, xs):
        tic = time.time()
        xs =  xs.to(device)

        xs_aug = []
        for _ in range(n_arms):
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
        loss, loss_rec, loss_joint, entropy, dist_c, d_qc, kl_cont, min_var_0, ll = model.loss(xs_rec, p_x, r_x, xs_aug, mu, logvar, qc, c, c_bin)
        loss.backward()
        solver.step()

        n_samples = len(xs)
        dt = time.time() - tic
        return {
            'loss': loss.item(),
            'loss_rec_0': loss_rec[0].item(),
            'loss_rec_1': loss_rec[1].item(),
            'loss_joint': loss_joint.item(),
            'throughput': n_samples / dt, # it/s
            'dt': dt * 1000.0  # ms
        }
    t0 = time.time()
    step = 0
    with wandb.init(entity='', project='switchvae', config=spec) as run:
        model.train()
        for ep in range(n_epochs):
            for i_batch, (xs, i_xs) in enumerate(train_loader):
                losses = train_step(model, solver, xs)
                step += 1
                run.log({
                    'step': step,
                    'epoch': ep,
                    'batch': i_batch,
                    **losses
                })
                if step % print_every == 0:
                    print(f"step: {step} | " + " | ".join(f"{k}: " + f"{v:.2f}" for k, v in losses.items()))
                print(f"step: {step} | " + " | ".join(f"{k}: " + f"{v:.2f}" for k, v in losses.items()))
        print(f"training time: {time.time() - t0:.2f}s")