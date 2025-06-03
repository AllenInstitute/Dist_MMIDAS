from typing import Any

import torch as th
from torch import nn

from nn_model import mixVAE_model

Params = dict[str, th.Tensor]
ModelSpec = dict[str, Any]

def module_params(model: nn.Module) -> Params:
    return model.state_dict()

def module_n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def params_is_equal(ps1: Params, ps2: Params) -> bool:
    return set(ps1.keys()) == set(ps2.keys()) and all(th.equal(ps1[k], ps2[k]) for k in ps1)

def make_mmidas(spec: ModelSpec) -> nn.Module:
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
