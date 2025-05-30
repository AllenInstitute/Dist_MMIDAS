from nn_model import mixVAE_model

def make_mmidas(spec):
    return mixVAE_model(
        input_dim=spec['input_dim'],
        fc_dim=spec['fc_dim'],
        n_categories=spec['n_categories'],
        state_dim=spec['state_dim'],
        lowD_dim=spec['lowD_dim'],
        x_drop=spec['x_drop'],
        s_drop=spec['s_drop'],
        n_arm=spec['n_arm'],
        lam=spec['lam'],
        lam_pc=spec['lam_pc'],
        tau=spec['tau'],
        beta=spec['beta'],
        hard=spec['hard'],
        variational=spec['variational'],
        device=spec['device'],
        eps=spec['eps'],
        ref_prior=spec['ref_prior'],
        momentum=spec['momentum'],
        loss_mode=spec['loss']
    ).to(spec['device'])
