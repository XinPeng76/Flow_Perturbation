import torch
import os
import numpy as np
from train_model_CGN import alphas_prod,n_dimensions,n_particles,ndim,device,num_steps,target_energy
from train_Var_CGN import (n_dimensions, n_particles, ndim, back_coeff,time_forward, time_backward)
from src.utils import remove_mean, clean_up
from utils import load_models_and_data_CGN
from src.MC import run_mcmc_and_save_CoM
def get_log_omega(xT, eps):
    x0 = ddpm_ode_heun(xT, time_backward) + back_coeff * eps

    forw_coeff = model_var(x0)
    eps_tilde = (xT - ddpm_ode_heun(x0, time_forward))/forw_coeff
    deltaSt = -0.5 * torch.sum(eps_tilde**2 - eps**2, dim=-1) - ndim * torch.log(forw_coeff).squeeze(-1) + ndim * np.log(back_coeff)
    uz = torch.sum(xT**2, dim=-1)/2.0  + 0.5*ndim*np.log(2*np.pi) 
    ux = target_energy.energy(x0).squeeze(1)
    log_omega = -ux + deltaSt + uz

    return log_omega, x0, ux

if __name__ == '__main__':
    if not os.path.exists('data'):
        os.makedirs('data')

    model_var, ddpm_ode_heun = load_models_and_data_CGN(ndim, num_steps, alphas_prod, n_particles, n_dimensions, back_coeff,load_var=True)
    sampN = 10000
    nmcmc = 60000
    xT_init = torch.randn(sampN, ndim).to(device)
    xT_init = remove_mean(xT_init, n_particles, n_dimensions)
    eps_init = torch.randn_like(xT_init)
    eps_init = remove_mean(eps_init, n_particles, n_dimensions)
    log_omega_init, x0_init, ux_init = get_log_omega(xT_init, eps_init)
    print(ux_init.min())
    samples_selected = ux_init < 2000
    xT = xT_init[samples_selected].clone()
    eps = eps_init[samples_selected].clone()
    x0 = x0_init[samples_selected].clone()
    log_omega = log_omega_init[samples_selected].clone()
    ux = ux_init[samples_selected].clone()
    print(xT.shape)
    run_mcmc_and_save_CoM(nmcmc, ux, x0, xT, eps, log_omega, n_particles, n_dimensions, device, back_coeff,get_log_omega, K_x=1, K_eps=1, if_eps=True,path='data/CGN-FP')
    clean_up()



