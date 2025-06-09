from typing import Any

import torch as th
from torch import nn
from torch.nn import ModuleList as mdl
import torch.nn.functional as F

from nn_model import mixVAE_model

Params = dict[str, th.Tensor]
MMIDASSpec = dict[str, Any]

class MMIDAS(nn.Module):
    def __init__(self, input_dim, fc_dim, n_categories, state_dim, lowD_dim, x_drop, s_drop, n_arm, lam, lam_pc,
                 tau, beta, hard, variational, device, eps, momentum, ref_prior, loss_mode):
        super(MMIDAS, self).__init__()
        self.input_dim = input_dim
        self.fc_dim = fc_dim
        self.state_dim = state_dim
        self.n_categories = n_categories
        self.x_dp = nn.Dropout(x_drop)
        self.s_dp = nn.Dropout(s_drop)
        self.hard = hard
        self.n_arm = n_arm
        self.lam = lam
        self.lam_pc = lam_pc
        self.tau = tau
        self.beta = beta
        self.varitional = variational
        self.eps = eps
        self.ref_prior = ref_prior
        self.momentum = momentum
        self.device = device
        self.loss_mode = loss_mode

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.fc1 = mdl([nn.Linear(input_dim, fc_dim) for i in range(n_arm)])
        self.fc2 = mdl([nn.Linear(fc_dim, fc_dim) for i in range(n_arm)])
        self.fc3 = mdl([nn.Linear(fc_dim, fc_dim) for i in range(n_arm)])
        self.fc4 = mdl([nn.Linear(fc_dim, fc_dim) for i in range(n_arm)])
        self.fc5 = mdl([nn.Linear(fc_dim, lowD_dim) for i in range(n_arm)])
        self.fcc = mdl([nn.Linear(lowD_dim, n_categories) for i in range(n_arm)])
        self.fc_mu = mdl([nn.Linear(lowD_dim + n_categories, state_dim) for i in range(n_arm)])
        self.fc_sigma = mdl([nn.Linear(lowD_dim + n_categories, state_dim) for i in range(n_arm)])
        self.fc6 = mdl([nn.Linear(state_dim + n_categories, lowD_dim) for i in range(n_arm)])
        self.fc7 = mdl([nn.Linear(lowD_dim, fc_dim) for i in range(n_arm)])
        self.fc8 = mdl([nn.Linear(fc_dim, fc_dim) for i in range(n_arm)])
        self.fc9 = mdl([nn.Linear(fc_dim, fc_dim) for i in range(n_arm)])
        self.fc10 = mdl([nn.Linear(fc_dim, fc_dim) for i in range(n_arm)])
        self.fc11 = mdl([nn.Linear(fc_dim, input_dim) for i in range(n_arm)])
        if loss_mode == 'ZINB':
            self.fc11_p = mdl([nn.Linear(fc_dim, input_dim) for i in range(n_arm)])
            self.fc11_r = mdl([nn.Linear(fc_dim, input_dim) for i in range(n_arm)])


        self.batch_l1 = mdl([nn.BatchNorm1d(num_features=fc_dim, eps=eps, momentum=momentum, affine=False) for i in range(n_arm)])
        self.batch_l2 = mdl([nn.BatchNorm1d(num_features=fc_dim, eps=eps, momentum=momentum, affine=False) for i in range(n_arm)])
        self.batch_l3 = mdl([nn.BatchNorm1d(num_features=fc_dim, eps=eps, momentum=momentum, affine=False) for i in range(n_arm)])
        self.batch_l4 = mdl([nn.BatchNorm1d(num_features=fc_dim, eps=eps, momentum=momentum, affine=False) for i in range(n_arm)])
        self.batch_l5 = mdl([nn.BatchNorm1d(num_features=lowD_dim, eps=eps, momentum=momentum, affine=False) for i in range(n_arm)])
        self.batch_s = mdl([nn.BatchNorm1d(num_features=state_dim, eps=eps, momentum=momentum, affine=False) for i in range(n_arm)])

        self.c_var_inv = [None] * 2
        self.stack_mean = [[] for a in range(2)]
        self.stack_var = [[] for a in range(2)]
        self.c_mean = [None] * 2
        self.c_var = [None] * 2

    def encoder(self, x, arm):
        x = self.batch_l1[arm](self.relu(self.fc1[arm](self.x_dp(x))))
        x = self.batch_l2[arm](self.relu(self.fc2[arm](x)))
        x = self.batch_l3[arm](self.relu(self.fc3[arm](x)))
        x = self.batch_l4[arm](self.relu(self.fc4[arm](x)))
        z = self.batch_l5[arm](self.relu(self.fc5[arm](x)))
        return z, F.softmax(self.fcc[arm](z), dim=-1)

    def intermed(self, x, arm):
        if self.varitional:
            return self.fc_mu[arm](x), self.sigmoid(self.fc_sigma[arm](x))
        else:
            return self.fc_mu[arm](x)


    def decoder(self, c, s, arm):
        s = self.s_dp(s)
        z = th.cat((c, s), dim=1)
        x = self.relu(self.fc6[arm](z))
        x = self.relu(self.fc7[arm](x))
        x = self.relu(self.fc8[arm](x))
        x = self.relu(self.fc9[arm](x))
        x = self.relu(self.fc10[arm](x))
        return self.relu(self.fc11[arm](x))
    
    def decoder_zinb(self, c, s, arm):
        s = self.s_dp(s)
        z = th.cat((c, s), dim=1)
        x = self.relu(self.fc6[arm](z))
        x = self.relu(self.fc7[arm](x))
        x = self.relu(self.fc8[arm](x))
        x = self.relu(self.fc9[arm](x))
        x = self.relu(self.fc10[arm](x))
        return self.relu(self.fc11[arm](x)), self.sigmoid(self.fc11_p[arm](x)), self.sigmoid(self.fc11_r[arm](x))

    def forward(self, x, temp, prior_c=[], eval=False, mask=None):
        recon_x = [None] * self.n_arm
        zinb_pi = [None] * self.n_arm
        zinb_r = [None] * self.n_arm
        p_x = [None] * self.n_arm
        s, c = [None] * self.n_arm, [None] * self.n_arm
        mu, log_var = [None] * self.n_arm, [None] * self.n_arm
        qc, alr_qc = [None] * self.n_arm, [None] * self.n_arm
        x_low, log_qc = [None] * self.n_arm, [None] * self.n_arm

        for arm in range(self.n_arm):
            x_low[arm], log_qc[arm] = self.encoder(x[arm], arm)

            if mask is not None:
                qc_tmp = F.softmax(log_qc[arm][:, mask] / self.tau, dim=-1)
                qc[arm] = torch.zeros((log_qc[arm].size(0), log_qc[arm].size(1))).to(self.device)

                qc[arm][:, mask] = qc_tmp
            else:
                qc[arm] = F.softmax(log_qc[arm] / self.tau, dim=-1)

            q_ = qc[arm].view(log_qc[arm].size(0), 1, self.n_categories)

            if eval:
                c[arm] = self.gumbel_softmax(q_, 1, self.n_categories, temp, hard=True, gumble_noise=False)
            else:
                c[arm] = self.gumbel_softmax(q_, 1, self.n_categories, temp, hard=self.hard)

            if self.ref_prior:
                y = th.cat((x_low[arm], prior_c), dim=1)
            else:
                y = th.cat((x_low[arm], c[arm]), dim=1)

            if self.varitional:
                mu[arm], var = self.intermed(y, arm)
                log_var[arm] = (var + self.eps).log()
                s[arm] = self.reparam_trick(mu[arm], log_var[arm])
            else:
                mu[arm] = self.intermed(y, arm)
                log_var[arm] = 0. * mu[arm]
                s[arm] = self.intermed(y, arm)
            
            if self.loss_mode == 'ZINB':
                recon_x[arm], zinb_pi[arm], zinb_r[arm] = self.decoder_zinb(c[arm], s[arm], arm)
            else:
                recon_x[arm] = self.decoder(c[arm], s[arm], arm)

        return recon_x, zinb_pi, zinb_r, x_low, qc, s, c, mu, log_var, log_qc

    def reparam_trick(self, mu, log_sigma):
        std = log_sigma.exp().sqrt()
        eps = th.rand_like(std).to(self.device)
        return eps.mul(std).add(mu)

    def sample_gumbel(self, shape):
        U = th.rand(shape).to(self.device)

        return -Variable(th.log(-th.log(U + self.eps) + self.eps))


    def gumbel_softmax_sample(self, phi, temperature):
        logits = (phi + self.eps).log() + self.sample_gumbel(phi.size())
        return F.softmax(logits / temperature, dim=-1)


    def gumbel_softmax(self, phi, latent_dim, categorical_dim, temperature, hard=False, gumble_noise=True):
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
        loss_indep, KLD_cont = [None] * self.n_arm, [None] * self.n_arm
        log_qz, l_rec = [None] * self.n_arm, [None] * self.n_arm
        var_qz, var_qz_inv = [None] * self.n_arm, [None] * self.n_arm
        mu_in, var_in = [None] * self.n_arm, [None] * self.n_arm
        mu_tmp, var_tmp = [None] * self.n_arm, [None] * self.n_arm
        loglikelihood = [None] * self.n_arm
        batch_size, n_cat = c[0].size()
        neg_joint_entropy, z_distance_rep, z_distance, dist_a = [], [], [], []

        for arm_a in range(self.n_arm):
            loglikelihood[arm_a] = F.mse_loss(recon_x[arm_a], x[arm_a], reduction='mean') + x[arm_a].size(0) * np.log(2 * np.pi)
            if self.loss_mode == 'MSE':
                l_rec[arm_a] = 0.5 * F.mse_loss(recon_x[arm_a], x[arm_a], reduction='sum') / (x[arm_a].size(0))
                rec_bin = th.where(recon_x[arm_a] > 0.1, 1., 0.)
                x_bin = th.where(x[arm_a] > 0.1, 1., 0.)
                l_rec[arm_a] += 0.5 * F.binary_cross_entropy(rec_bin, x_bin)
            elif self.loss_mode == 'ZINB':
                l_rec[arm_a] = zinb_loss(recon_x[arm_a], p_x[arm_a], r_x[arm_a], x[arm_a])

            if self.varitional:
                KLD_cont[arm_a] = (-0.5 * th.mean(1 + log_sigma[arm_a] - mu[arm_a].pow(2) - log_sigma[arm_a].exp(), dim=0)).sum()
                loss_indep[arm_a] = l_rec[arm_a] + self.beta * KLD_cont[arm_a]
            else:
                loss_indep[arm_a] = l_rec[arm_a]
                KLD_cont[arm_a] = [0.]

            log_qz[0] = torch.log(qc[arm_a] + self.eps)
            var_qz0 = qc[arm_a].var(0)

            var_qz_inv[0] = (1 / (var_qz0 + self.eps)).repeat(qc[arm_a].size(0), 1).sqrt()

            for arm_b in range(arm_a + 1, self.n_arm):
                log_qz[1] = th.log(qc[arm_b] + self.eps)
                tmp_entropy = (th.sum(qc[arm_a] * log_qz[0], dim=-1)).mean() + \
                              (th.sum(qc[arm_b] * log_qz[1], dim=-1)).mean()
                neg_joint_entropy.append(tmp_entropy)
                # var = qc[arm_b].var(0)
                var_qz1 = qc[arm_b].var(0)
                var_qz_inv[1] = (1 / (var_qz1 + self.eps)).repeat(qc[arm_b].size(0), 1).sqrt()

                # distance between z_1 and z_2 i.e., ||z_1 - z_2||^2
                # Euclidean distance
                z_distance_rep.append((th.norm((c[arm_a] - c[arm_b]), p=2, dim=1).pow(2)).mean())
                z_distance.append((th.norm((log_qz[0] * var_qz_inv[0]) - (log_qz[1] * var_qz_inv[1]), p=2, dim=1).pow(2)).mean())

            if self.ref_prior:
                n_comb = max(self.n_arm * (self.n_arm + 1) / 2, 1)
                scaler = self.n_arm
                # distance between z_1 and z_2 i.e., ||z_1 - z_2||^2
                # Euclidean distance
                z_distance_rep.append((th.norm((c[arm_a] - prior_c), p=2, dim=1).pow(2)).mean())
                tmp_entropy = (th.sum(qc[arm_a] * log_qz[0], dim=-1)).mean()
                neg_joint_entropy.append(tmp_entropy)
                qc_bin = self.gumbel_softmax(qc[arm_a], 1, self.n_categories, 1, hard=True, gumble_noise=False)
                z_distance.append(self.lam_pc * F.binary_cross_entropy(qc_bin, prior_c))
            else:
                n_comb = max(self.n_arm * (self.n_arm - 1) / 2, 1)
                scaler = max((self.n_arm - 1), 1)


        loss_joint = self.lam * sum(z_distance) + sum(neg_joint_entropy) + n_comb * ((n_cat / 2) * (np.log(2 * np.pi)) - 0.5 * np.log(2 * self.lam))

        loss = scaler * sum(loss_indep) + loss_joint

        return loss, l_rec, loss_joint, sum(neg_joint_entropy) / n_comb, sum(z_distance) / n_comb, sum(z_distance_rep) / n_comb, KLD_cont, var_qz0.min(), loglikelihood


def zinb_loss(rec_x, x_p, x_r, X, eps=1e-6):
    X_dim = X.size(-1)
    k = X.exp() - 1. #logp(count) -->  (count)

    # extracting r,p, and z from the concatenated vactor.
    # eps added for stability.
    r = rec_x + eps # zinb_params[:, :X_dim] + eps
    p = (1 - eps)*(x_p + eps) # (1 - eps)*(zinb_params[:, X_dim:2*X_dim] + eps)
    z = (1 - eps)*(x_r + eps) # (1 - eps)*(zinb_params[:, 2*X_dim:] + eps)

    mask_nonzeros = ([X > 0])[0].to(th.float32)
    loss_zero_counts = (mask_nonzeros-1) * (z + (1-z) * (1-p).pow(r)).log()
    # log of zinb for non-negative terms, excluding x! term
    loss_nonzero_counts = mask_nonzeros * (-(k + r).lgamma() + r.lgamma() - k*p.log() - r*(1-p).log() - (1-z).log())

    l_zinb = (loss_zero_counts + loss_nonzero_counts).mean()

    return l_zinb

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
):
    return {
        'input_dim': input_dim,
        'fc_dim': fc_dim,
        'lowD_dim': lowD_dim,
        'state_dim': state_dim,
        'n_categories': n_categories,
        'n_arms': n_arms,
        'temp': temp,
        'eps': eps,
        'is_ref_prior': is_ref_prior,
        'x_drop': x_drop,
        's_drop': s_drop,
        'lam': lam,
        'lam_pc': lam_pc,
        'tau': tau,
        'beta': beta,
        'is_hard': is_hard,
        'is_variational': is_variational,
        'momentum': 0.01,
        'c_prior': c_prior,
        'c_onehot': c_onehot,
        'loss': 'MSE',
        'type': "MMIDASSPec"
    }

def module_params(model: nn.Module) -> Params:
    return model.state_dict()

def module_n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def params_is_equal(ps1: Params, ps2: Params) -> bool:
    return set(ps1.keys()) == set(ps2.keys()) and all(th.equal(ps1[k], ps2[k]) for k in ps1)

def make_mmidas(spec: MMIDASSpec) -> nn.Module:
    return mixVAE_model(
        input_dim=spec['input_dim'],
        fc_dim=spec['fc_dim'],
        n_categories=spec['n_categories'],
        state_dim=spec['state_dim'],
        lowD_dim=spec['lowD_dim'],
        x_drop=spec['x_drop'],
        s_drop=spec['s_drop'],
        n_arm=spec['n_arms'],
        lam=spec['lam'],
        lam_pc=spec['lam_pc'],
        tau=spec['tau'],
        beta=spec['beta'],
        hard=spec['is_hard'],
        variational=spec['is_variational'],
        device=spec['device'],
        eps=spec['eps'],
        ref_prior=spec['is_ref_prior'],
        momentum=spec['momentum'],
        loss_mode=spec['loss']
    ).to(spec['device'])

def make_mmidas2(spec: MMIDASSpec) -> nn.Module:
    raise NotImplementedError("MMIDAS2 is not implemented yet")
    return MMIDAS(spec)