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

def mean(xs):
    return sum(xs) / len(xs)

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
        A = mspec_lookup(self.spec, "n_arms")
        K = mspec_lookup(self.spec, "n_categories")
        eps = mspec_lookup(self.spec, "eps")
        tau = mspec_lookup(self.spec, "tau")
        loss_fn = mspec_lookup(self.spec, "loss_fn")
        is_hard = mspec_lookup(self.spec, "is_hard")
        is_ref_prior = mspec_lookup(self.spec, "is_ref_prior")
        is_variational = mspec_lookup(self.spec, "is_variational")

        device = mspec_lookup(self.spec, "device")

        x_rec = [None] * A
        zinb_pi = [None] * A
        zinb_r = [None] * A
        s, c = [None] * A, [None] * A
        s_mean, s_logvar = [None] * A, [None] * A
        qc = [None] * A
        x_low, log_qc = [None] * A, [None] * A

        for a in range(A):
            x_low[a], log_qc[a] = self.encoder(x[a], a)

            if mask is not None:
                qc_tmp = F.softmax(log_qc[a][:, mask] / tau, dim=-1)
                qc[a] = th.zeros((log_qc[a].size(0), log_qc[a].size(1))).to(device)

                qc[a][:, mask] = qc_tmp
            else:
                qc[a] = F.softmax(log_qc[a] / tau, dim=-1)

            q_ = qc[a].view(len(log_qc[a]), 1, K)

            if eval:
                c[a] = self.gumbel_softmax(
                    q_, 1, K, temp, hard=True, gumble_noise=False
                )
            else:
                c[a] = self.gumbel_softmax(q_, 1, K, temp, hard=is_hard)

            if is_ref_prior:
                y = th.cat((x_low[a], prior_c), dim=1)
            else:
                y = th.cat((x_low[a], c[a]), dim=1)

            if is_variational:
                s_mean[a], var = self.intermed(y, a)
                s_logvar[a] = (var + eps).log()
                s[a] = self.reparam_trick(s_mean[a], s_logvar[a])
            else:
                s_mean[a] = self.intermed(y, a)
                s_logvar[a] = 0.0 * s_mean[a]
                s[a] = self.intermed(y, a)

            if loss_fn == "ZINB":
                x_rec[a], zinb_pi[a], zinb_r[a] = self.decoder_zinb(c[a], s[a], a)
            else:
                x_rec[a] = self.decoder(c[a], s[a], a)

        return x_rec, zinb_pi, zinb_r, x_low, qc, s, c, s_mean, s_logvar, log_qc

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

    def loss(self, x_rec, p_x, r_x, x, s_mean, s_logvar, qc, c, prior_c=[]):
        A = mspec_lookup(self.spec, "n_arms")
        K = mspec_lookup(self.spec, "n_categories")
        beta = mspec_lookup(self.spec, "beta")
        eps = mspec_lookup(self.spec, "eps")
        lam = mspec_lookup(self.spec, "lam")
        lam_pc = mspec_lookup(self.spec, "lam_pc")
        loss_fn = mspec_lookup(self.spec, "loss_fn")
        is_ref_prior = mspec_lookup(self.spec, "is_ref_prior")
        is_variational = mspec_lookup(self.spec, "is_variational")

        loglikelihood = []
        l_rec = []
        loss_indep = []
        kl_cont = []
    
        entropy = [] # negative joint entropy between qc of two arms
        qc_l2_dist = [] # Euclidean distance between z_1 and z_2 i.e., ||z_1 - z_2||^2
        qc_simplex_dist = []
        for a in range(A):
            _loglikelihood = F.mse_loss(x_rec[a], x[a], reduction="mean") + len(x[a]) * np.log(2 * np.pi)
            if loss_fn == "MSE":
                _l_rec = (0.5 * F.mse_loss(x_rec[a], x[a], reduction="sum") / len(x[a])) + (0.5 * F.binary_cross_entropy((x_rec[a] > 0.1).float(), (x[a] > 0.1).float()))
            elif loss_fn == "ZINB":
                _l_rec = zinb_loss(x_rec[a], p_x[a], r_x[a], x[a], eps=eps)
            else:
                raise NotImplementedError(f"Unknown loss function: {loss_fn}")

            if is_variational:
                _kl_cont = (-0.5 * th.mean(1 + s_logvar[a] - s_mean[a].pow(2) - s_logvar[a].exp(), dim=0)).sum()
                _loss_indep = _l_rec + beta * _kl_cont
            else:
                _kl_cont = [0.0]
                _loss_indep = _l_rec

            loglikelihood.append(_loglikelihood)
            l_rec.append(_l_rec)
            kl_cont.append(_kl_cont)
            loss_indep.append(_loss_indep)

            log_qc_a = th.log(qc[a] + eps)
            var_qc_a = qc[a].var(0)
            var_qc_a_inv = (1 / (var_qc_a + eps)).repeat(len(qc[a]), 1).sqrt()
            for b in range(a + 1, A):
                log_qc_b = th.log(qc[b] + eps)
                var_qc_b = qc[b].var(0)
                var_qc_b_inv = ((1 / (var_qc_b + eps)).repeat(len(qc[b]), 1).sqrt())

                _entropy = (th.sum(qc[a] * log_qc_a, dim=-1)).mean() + (th.sum(qc[b] * log_qc_b, dim=-1)).mean()
                _qc_l2_dist = (th.norm((c[a] - c[b]), p=2, dim=1).pow(2)).mean()
                _qc_simplex_dist = (th.norm((log_qc_a * var_qc_a_inv) - (log_qc_b * var_qc_b_inv), p=2, dim=1).pow(2)).mean()

                entropy.append(_entropy)
                qc_l2_dist.append(_qc_l2_dist)
                qc_simplex_dist.append(_qc_simplex_dist)

            if is_ref_prior:
                print("warning: enabling a prior is untested!")
                n_comb = max(A * (A + 1) / 2, 1)
                scaler = A
                qc_l2_dist.append((th.norm((c[a] - prior_c), p=2, dim=1).pow(2)).mean())
                tmp_entropy = (th.sum(qc[a] * log_qc_a, dim=-1)).mean()
                entropy.append(tmp_entropy)
                qc_bin = self.gumbel_softmax(qc[a], 1, K, 1, hard=True, gumble_noise=False)
                qc_simplex_dist.append(lam_pc * F.binary_cross_entropy(qc_bin, prior_c))

        n_comb = max(A * (A - 1) / 2, 1)
        scaler = max((A - 1), 1)
        loss_joint = (
            lam * sum(qc_simplex_dist)
            + sum(entropy)
            + n_comb * ((K / 2) * (np.log(2 * np.pi)) - 0.5 * np.log(2 * lam))
        )

        loss = scaler * sum(loss_indep) + loss_joint

        return (
            loss,
            l_rec,
            loss_joint,
            mean(entropy),
            mean(qc_simplex_dist),
            mean(qc_l2_dist),
            kl_cont,
            var_qc_a.min(),
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
