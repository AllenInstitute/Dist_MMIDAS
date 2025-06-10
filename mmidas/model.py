from typing import Any

import numpy as np
import torch as th
from torch import nn
from torch.nn import ModuleList as mdl
from torch.autograd import Variable
import torch.nn.functional as F

from nn_model import mixVAE_model

Params = dict[str, th.Tensor]
MMIDASSpec = dict[str, Any]


class MMIDAS(nn.Module):
    def __init__(self, spec):
        super(MMIDAS, self).__init__()

        self.spec = spec

        D = mspec_lookup(spec, "input_dim")
        H = mspec_lookup(spec, "fc_dim")
        L = mspec_lookup(spec, "lowD_dim")
        Z = mspec_lookup(spec, "state_dim")
        A = mspec_lookup(spec, "n_arms")
        K = mspec_lookup(spec, "n_categories")
        eps = mspec_lookup(spec, "eps")
        momentum = mspec_lookup(spec, "momentum")
        x_drop = mspec_lookup(spec, "x_drop")
        s_drop = mspec_lookup(spec, "s_drop")
        loss_fn = mspec_lookup(spec, "loss_fn")

        self.x_dp = nn.Dropout(x_drop)
        self.s_dp = nn.Dropout(s_drop)

        self.fc1 = mdl([nn.Linear(D, H) for _ in range(A)])
        self.fc2 = mdl([nn.Linear(H, H) for _ in range(A)])
        self.fc3 = mdl([nn.Linear(H, H) for _ in range(A)])
        self.fc4 = mdl([nn.Linear(H, H) for _ in range(A)])
        self.fc5 = mdl([nn.Linear(H, L) for _ in range(A)])
        self.fcc = mdl([nn.Linear(L, K) for _ in range(A)])
        self.fc_mu = mdl([nn.Linear(L + K, Z) for _ in range(A)])
        self.fc_sigma = mdl([nn.Linear(L + K, Z) for _ in range(A)])
        self.fc6 = mdl([nn.Linear(Z + K, L) for _ in range(A)])
        self.fc7 = mdl([nn.Linear(L, H) for _ in range(A)])
        self.fc8 = mdl([nn.Linear(H, H) for _ in range(A)])
        self.fc9 = mdl([nn.Linear(H, H) for _ in range(A)])
        self.fc10 = mdl([nn.Linear(H, H) for _ in range(A)])
        self.fc11 = mdl([nn.Linear(H, D) for _ in range(A)])
        if loss_fn == "ZINB":
            self.fc11_p = mdl([nn.Linear(H, D) for _ in range(A)])
            self.fc11_r = mdl([nn.Linear(H, D) for _ in range(A)])

        self.norm1 = mdl([nn.BatchNorm1d(num_features=H, eps=eps, momentum=momentum, affine=False) for _ in range(A)]) 
        self.norm2 = mdl([nn.BatchNorm1d(num_features=H, eps=eps, momentum=momentum, affine=False) for _ in range(A)])
        self.norm3 = mdl([nn.BatchNorm1d(num_features=H, eps=eps, momentum=momentum, affine=False) for _ in range(A)])
        self.norm4 = mdl([nn.BatchNorm1d(num_features=H, eps=eps, momentum=momentum, affine=False) for _ in range(A)])
        self.norm5 = mdl([nn.BatchNorm1d(num_features=L, eps=eps, momentum=momentum, affine=False) for _ in range(A)])

        self.c_var_inv = [None] * 2
        self.stack_mean = [[] for a in range(2)]
        self.stack_var = [[] for a in range(2)]
        self.c_mean = [None] * 2
        self.c_var = [None] * 2

    def encoder(self, x, a):
        x = self.norm1[a](F.relu(self.fc1[a](self.x_dp(x))))
        x = self.norm2[a](F.relu(self.fc2[a](x)))
        x = self.norm3[a](F.relu(self.fc3[a](x)))
        x = self.norm4[a](F.relu(self.fc4[a](x)))
        z = self.norm5[a](F.relu(self.fc5[a](x)))
        return z, F.softmax(self.fcc[a](z), dim=-1)

    def intermed(self, x, a):
        is_variational = mspec_lookup(self.spec, "is_variational")
        if is_variational:
            return self.fc_mu[a](x), F.sigmoid(self.fc_sigma[a](x))
        else:
            return self.fc_mu[a](x)

    def decoder(self, c, s, arm):
        s = self.s_dp(s)
        z = th.cat((c, s), dim=1)
        x = F.relu(self.fc6[arm](z))
        x = F.relu(self.fc7[arm](x))
        x = F.relu(self.fc8[arm](x))
        x = F.relu(self.fc9[arm](x))
        x = F.relu(self.fc10[arm](x))
        return F.relu(self.fc11[arm](x))

    def decoder_zinb(self, c, s, arm):
        s = self.s_dp(s)
        z = th.cat((c, s), dim=1)
        x = F.relu(self.fc6[arm](z))
        x = F.relu(self.fc7[arm](x))
        x = F.relu(self.fc8[arm](x))
        x = F.relu(self.fc9[arm](x))
        x = F.relu(self.fc10[arm](x))
        return (
            F.relu(self.fc11[arm](x)),
            F.sigmoid(self.fc11_p[arm](x)),
            F.sigmoid(self.fc11_r[arm](x)),
        )

    def forward(self, x, temp, prior_c=[], eval=False, mask=None):
        n_arms = mspec_lookup(self.spec, "n_arms")
        n_categories = mspec_lookup(self.spec, "n_categories")
        eps = mspec_lookup(self.spec, "eps")
        tau = mspec_lookup(self.spec, "tau")
        loss_fn = mspec_lookup(self.spec, "loss_fn")
        is_hard = mspec_lookup(self.spec, "is_hard")
        is_ref_prior = mspec_lookup(self.spec, "is_ref_prior")
        is_variational = mspec_lookup(self.spec, "is_variational")

        device = mspec_lookup(self.spec, "device")

        recon_x = [None] * n_arms
        zinb_pi = [None] * n_arms
        zinb_r = [None] * n_arms
        s, c = [None] * n_arms, [None] * n_arms
        mu, log_var = [None] * n_arms, [None] * n_arms
        qc = [None] * n_arms
        x_low, log_qc = [None] * n_arms, [None] * n_arms

        for a in range(n_arms):
            x_low[a], log_qc[a] = self.encoder(x[a], a)

            if mask is not None:
                qc_tmp = F.softmax(log_qc[a][:, mask] / tau, dim=-1)
                qc[a] = th.zeros((log_qc[a].size(0), log_qc[a].size(1))).to(device)

                qc[a][:, mask] = qc_tmp
            else:
                qc[a] = F.softmax(log_qc[a] / tau, dim=-1)

            q_ = qc[a].view(len(log_qc[a]), 1, n_categories)

            if eval:
                c[a] = self.gumbel_softmax(
                    q_, 1, n_categories, temp, hard=True, gumble_noise=False
                )
            else:
                c[a] = self.gumbel_softmax(q_, 1, n_categories, temp, hard=is_hard)

            if is_ref_prior:
                y = th.cat((x_low[a], prior_c), dim=1)
            else:
                y = th.cat((x_low[a], c[a]), dim=1)

            if is_variational:
                mu[a], var = self.intermed(y, a)
                log_var[a] = (var + eps).log()
                s[a] = self.reparam_trick(mu[a], log_var[a])
            else:
                mu[a] = self.intermed(y, a)
                log_var[a] = 0.0 * mu[a]
                s[a] = self.intermed(y, a)

            if loss_fn == "ZINB":
                recon_x[a], zinb_pi[a], zinb_r[a] = self.decoder_zinb(c[a], s[a], a)
            else:
                recon_x[a] = self.decoder(c[a], s[a], a)

        return recon_x, zinb_pi, zinb_r, x_low, qc, s, c, mu, log_var, log_qc

    def reparam_trick(self, mu, log_sigma):
        device = mspec_lookup(self.spec, "device")

        std = log_sigma.exp().sqrt()
        eps = th.rand_like(std).to(device)
        return eps.mul(std).add(mu)

    def sample_gumbel(self, shape):
        device = mspec_lookup(self.spec, "device")

        eps = mspec_lookup(self.spec, "eps")
        U = th.rand(shape).to(device)

        return -Variable(th.log(-th.log(U + eps) + eps))

    def gumbel_softmax_sample(self, phi, temperature):
        eps = mspec_lookup(self.spec, "eps")
        logits = (phi + eps).log() + self.sample_gumbel(phi.size())
        return F.softmax(logits / temperature, dim=-1)

    def gumbel_softmax(
        self,
        phi,
        latent_dim,
        categorical_dim,
        temperature,
        hard=False,
        gumble_noise=True,
    ):
        if gumble_noise:
            y = self.gumbel_softmax_sample(phi, temperature)
        else:
            y = phi

        if not hard:
            return y.view(-1, latent_dim * categorical_dim)
        else:
            shape = y.size()
            _, ind = y.max(dim=-1)
            y_hard = th.zeros_like(y).view(-1, shape[-1])
            y_hard.scatter_(1, ind.view(-1, 1), 1)
            y_hard = y_hard.view(*shape)
            y_hard = (y_hard - y).detach() + y
            return y_hard.view(-1, latent_dim * categorical_dim)

    def loss(self, recon_x, p_x, r_x, x, mu, log_sigma, qc, c, prior_c=[]):
        A = mspec_lookup(self.spec, "n_arms")
        K = mspec_lookup(self.spec, "n_categories")
        beta = mspec_lookup(self.spec, "beta")
        eps = mspec_lookup(self.spec, "eps")
        lam = mspec_lookup(self.spec, "lam")
        lam_pc = mspec_lookup(self.spec, "lam_pc")
        loss_fn = mspec_lookup(self.spec, "loss_fn")
        is_ref_prior = mspec_lookup(self.spec, "is_ref_prior")
        is_variational = mspec_lookup(self.spec, "is_variational")

        loss_indep, kl_cont = [None] * A, [None] * A
        log_qz, l_rec = [None] * A, [None] * A
        var_qz_inv = [None] * A
        loglikelihood = [None] * A
        neg_joint_entropy, z_distance_rep, z_distance = [], [], []

        for a in range(A):
            loglikelihood[a] = F.mse_loss(recon_x[a], x[a], reduction="mean") + x[a].size(0) * np.log(2 * np.pi)
            if loss_fn == "MSE":
                l_rec[a] = (0.5 * F.mse_loss(recon_x[a], x[a], reduction="sum") / (x[a].size(0)))
                rec_bin = th.where(recon_x[a] > 0.1, 1.0, 0.0)
                x_bin = th.where(x[a] > 0.1, 1.0, 0.0)
                l_rec[a] += 0.5 * F.binary_cross_entropy(rec_bin, x_bin)
            elif loss_fn == "ZINB":
                l_rec[a] = zinb_loss(
                    recon_x[a], p_x[a], r_x[a], x[a]
                )
            else:
                raise NotImplementedError(f"Unknown loss function: {loss_fn}")

            if is_variational:
                kl_cont[a] = (-0.5 * th.mean(1 + log_sigma[a] - mu[a].pow(2) - log_sigma[a].exp(), dim=0)).sum()
                loss_indep[a] = l_rec[a] + beta * kl_cont[a]
            else:
                loss_indep[a] = l_rec[a]
                kl_cont[a] = [0.0]

            log_qz[0] = th.log(qc[a] + eps)
            var_qz0 = qc[a].var(0)

            var_qz_inv[0] = (1 / (var_qz0 + eps)).repeat(qc[a].size(0), 1).sqrt()

            for b in range(a + 1, A):
                log_qz[1] = th.log(qc[b] + eps)
                tmp_entropy = (th.sum(qc[a] * log_qz[0], dim=-1)).mean() + (
                    th.sum(qc[b] * log_qz[1], dim=-1)
                ).mean()
                neg_joint_entropy.append(tmp_entropy)
                # var = qc[arm_b].var(0)
                var_qz1 = qc[b].var(0)
                var_qz_inv[1] = (
                    (1 / (var_qz1 + eps)).repeat(qc[b].size(0), 1).sqrt()
                )

                # distance between z_1 and z_2 i.e., ||z_1 - z_2||^2
                # Euclidean distance
                z_distance_rep.append(
                    (th.norm((c[a] - c[b]), p=2, dim=1).pow(2)).mean()
                )
                z_distance.append((th.norm((log_qz[0] * var_qz_inv[0]) - (log_qz[1] * var_qz_inv[1]), p=2, dim=1).pow(2)).mean())

            if is_ref_prior:
                n_comb = max(A * (A + 1) / 2, 1)
                scaler = A
                # distance between z_1 and z_2 i.e., ||z_1 - z_2||^2
                # Euclidean distance
                z_distance_rep.append((th.norm((c[a] - prior_c), p=2, dim=1).pow(2)).mean())
                tmp_entropy = (th.sum(qc[a] * log_qz[0], dim=-1)).mean()
                neg_joint_entropy.append(tmp_entropy)
                qc_bin = self.gumbel_softmax(qc[a], 1, K, 1, hard=True, gumble_noise=False)
                z_distance.append(lam_pc * F.binary_cross_entropy(qc_bin, prior_c))
            else:
                n_comb = max(A * (A - 1) / 2, 1)
                scaler = max((A - 1), 1)

        loss_joint = (
            lam * sum(z_distance)
            + sum(neg_joint_entropy)
            + n_comb * ((K / 2) * (np.log(2 * np.pi)) - 0.5 * np.log(2 * lam))
        )

        loss = scaler * sum(loss_indep) + loss_joint

        return (
            loss,
            l_rec,
            loss_joint,
            sum(neg_joint_entropy) / n_comb,
            sum(z_distance) / n_comb,
            sum(z_distance_rep) / n_comb,
            kl_cont,
            var_qz0.min(),
            loglikelihood,
        )


def zinb_loss(rec_x, x_p, x_r, X, eps=1e-6):
    X_dim = X.size(-1)
    k = X.exp() - 1.0  # logp(count) -->  (count)

    # extracting r,p, and z from the concatenated vactor.
    # eps added for stability.
    r = rec_x + eps  # zinb_params[:, :X_dim] + eps
    p = (1 - eps) * (x_p + eps)  # (1 - eps)*(zinb_params[:, X_dim:2*X_dim] + eps)
    z = (1 - eps) * (x_r + eps)  # (1 - eps)*(zinb_params[:, 2*X_dim:] + eps)

    mask_nonzeros = ([X > 0])[0].to(th.float32)
    loss_zero_counts = (mask_nonzeros - 1) * (z + (1 - z) * (1 - p).pow(r)).log()
    # log of zinb for non-negative terms, excluding x! term
    loss_nonzero_counts = mask_nonzeros * (
        -(k + r).lgamma() + r.lgamma() - k * p.log() - r * (1 - p).log() - (1 - z).log()
    )

    l_zinb = (loss_zero_counts + loss_nonzero_counts).mean()

    return l_zinb


# TODO: remove device
def make_mspec(
    input_dim: int = 5032,
    fc_dim: int = 100,
    lowD_dim: int = 10,
    state_dim: int = 2,
    n_categories: int = 120,
    n_arms: int = 2,
    temp: float = 1.0,
    eps: float = 1e-8,
    is_ref_prior: bool = False,
    x_drop: float = 0.5,
    s_drop: float = 0.2,
    lam: float = 1.0,
    lam_pc: float = 1.0,
    tau: float = 0.005,
    beta: float = 1.0,
    is_hard: bool = False,
    is_variational: bool = True,
    c_prior: float = 0.0,
    c_onehot: float = 0.0,
    device: str = "cpu",
):
    return {
        "input_dim": input_dim,
        "fc_dim": fc_dim,
        "lowD_dim": lowD_dim,
        "state_dim": state_dim,
        "n_categories": n_categories,
        "n_arms": n_arms,
        "temp": temp,
        "eps": eps,
        "is_ref_prior": is_ref_prior,
        "x_drop": x_drop,
        "s_drop": s_drop,
        "lam": lam,
        "lam_pc": lam_pc,
        "tau": tau,
        "beta": beta,
        "is_hard": is_hard,
        "is_variational": is_variational,
        "momentum": 0.01,
        "c_prior": c_prior,
        "c_onehot": c_onehot,
        "loss_fn": "MSE",
        "device": device,
        "type": "MMIDASSPec",
    }


def mspec_lookup(spec: MMIDASSpec, key: str) -> Any:
    return spec[key]


def module_params(model: nn.Module) -> Params:
    return model.state_dict()


def module_n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def params_is_equal(ps1: Params, ps2: Params) -> bool:
    return set(ps1.keys()) == set(ps2.keys()) and all(
        th.equal(ps1[k], ps2[k]) for k in ps1
    )


def make_mmidas(spec: MMIDASSpec) -> nn.Module:
    return MMIDAS(spec)


def _make_mmidas(spec: MMIDASSpec) -> nn.Module:
    return mixVAE_model(
        input_dim=mspec_lookup(spec, "input_dim"),
        fc_dim=mspec_lookup(spec, "fc_dim"),
        n_categories=mspec_lookup(spec, "n_categories"),
        state_dim=mspec_lookup(spec, "state_dim"),
        lowD_dim=mspec_lookup(spec, "lowD_dim"),
        x_drop=mspec_lookup(spec, "x_drop"),
        s_drop=mspec_lookup(spec, "s_drop"),
        n_arm=mspec_lookup(spec, "n_arms"),
        lam=mspec_lookup(spec, "lam"),
        lam_pc=mspec_lookup(spec, "lam_pc"),
        tau=mspec_lookup(spec, "tau"),
        beta=mspec_lookup(spec, "beta"),
        hard=mspec_lookup(spec, "is_hard"),
        variational=mspec_lookup(spec, "is_variational"),
        device=mspec_lookup(spec, "device"),
        eps=mspec_lookup(spec, "eps"),
        ref_prior=mspec_lookup(spec, "is_ref_prior"),
        momentum=mspec_lookup(spec, "momentum"),
        loss_mode=mspec_lookup(spec, "loss_fn"),
    )
