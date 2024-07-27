import torch
import numpy as np
import os
from src.common import MLP, MLP_var, MLP_nonorm
    
def load_models_and_data_CGN(ndim, num_steps, alphas_prod, n_particles, n_dimensions, back_coeff,load_var=False):
    from src.DDPM import interpolate_parameters, DDPMSamplerCoM
    #from utils import DDPMSamplerCoM
    #from bgmol.datasets import ChignolinOBC2PT
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = MLP(ndim=ndim, hidden_size=2048, hidden_layers=12).to(device)
    if os.path.exists('models/CGN/model_RotAug_LowLR.pth'):
        model.load_state_dict(torch.load('models/CGN/model_RotAug_LowLR.pth', map_location=device))
    else:
        raise OSError('No model found, please train the model first!')
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    st, sigma_t, st_derivative, sigma_t_derivative = interpolate_parameters(num_steps, alphas_prod)
    Sampler = DDPMSamplerCoM(model, st, st_derivative, sigma_t_derivative, n_particles, n_dimensions, device)

    ddpm_ode_heun = Sampler.exact_dynamics_heun
    heun_torch = Sampler.heun_torch
    score_function_rearange = Sampler.score_function_rearange
    score_function_1element = Sampler.score_function_1element

    if load_var:
        from src.common import MLP_var
        model_var = MLP_var(ndim=ndim).to(device)
        if os.path.exists(f'models/CGN/model_var_{back_coeff}.pt'):
            model_var.load_state_dict(torch.load(f'models/CGN/model_var_{back_coeff}.pt', map_location=device))
        else:
            raise OSError(f'No model_var_{back_coeff} found, please train the model_var first!')
        model_var.eval()
        for param in model_var.parameters():
            param.requires_grad = False
        return model_var, ddpm_ode_heun
    
    return heun_torch, score_function_rearange, score_function_1element

def load_models_and_data_GMM(ndim, device, back_coeff, sig_data, load_var=False):
    from src.EDM import EMDSampler
    # Load the main model
    model = MLP_nonorm(ndim=ndim, hidden_size=2000, hidden_layers=10, emb_size=80).to(device)
    if os.path.exists(f'models/GMM/{ndim}-d-model.pth'):
        model.load_state_dict(torch.load(f'models/GMM/{ndim}-d-model.pth', map_location=device))
    else:
        raise OSError('No model found, please train the model first!')
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    # Create the sampler
    Sampler = EMDSampler(model, sig_data, device)
    exact_dynamics_heun = Sampler.exact_dynamics_heun
    heun_torch = Sampler.heun_torch
    score_function_rearange = Sampler.score_function_rearange
    score_function_1element = Sampler.score_function_1element
    if load_var:
        # Load the variance model
        model_var = MLP_var(ndim=ndim).to(device)
        model_var.load_state_dict(torch.load(f'models/GMM/model_var_{back_coeff}.pt', map_location=device))
        model_var.eval()
        for param in model_var.parameters():
            param.requires_grad = False
        return model_var, exact_dynamics_heun
    return heun_torch, score_function_rearange, score_function_1element
    