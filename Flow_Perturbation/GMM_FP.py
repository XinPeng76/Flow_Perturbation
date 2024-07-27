
import numpy as np
import torch
import os
from train_model_GMM import ndim,device,Nwellinfo,sig_data
from train_Var_GMM import back_coeff,epsilon,tmax
from src.GMM_distribution import get_energy_device
from utils import load_models_and_data_GMM
from src.MC import run_mcmc_and_save
def get_log_omega(xT, eps):
    x0 = exact_dynamics_heun(epsilon, tmax, xT) + back_coeff * eps
    # forward dynamics
    # xT = exact_dynamics_heun(tmax, 0, x0, mus0, covs0) + forw_coeff * eps_tilde
    forw_coeff = model_var(x0)
    eps_tilde = (xT - exact_dynamics_heun(tmax, epsilon, x0))/forw_coeff
    deltaSt = -0.5 * torch.sum(eps_tilde**2 - eps**2, dim=-1) - ndim * torch.log(forw_coeff).squeeze(-1) + ndim * np.log(back_coeff)
    uz = torch.sum(xT**2/tmax**2, dim=-1)/2.0  + 0.5*ndim*np.log(2*np.pi) + ndim*np.log(tmax)
    ux = get_energy_device(x0, Nwellinfo)
    log_omega = -ux + deltaSt + uz

    return log_omega, x0, ux

if __name__ == '__main__':
    if not os.path.exists('NWell_MCMC1000'):
        os.makedirs('NWell_MCMC1000')
    model_var, exact_dynamics_heun = load_models_and_data_GMM(ndim, device, back_coeff, sig_data, load_var=True)
    # generate starting point for the MCMC move
    # we need xT_init, eps_init, log_omega_init
    # do the backward-forward dynamics, get deltaSt, determine forward coeff
    sampN = 1000
    nmcmc = 40000
    lambdaT = tmax**2
    xT_init = np.sqrt(lambdaT) * torch.randn(sampN, ndim).to(device)
    eps_init = torch.randn_like(xT_init)

    log_omega_init, x0_init,ux_init = get_log_omega(xT_init, eps_init)

    xT = xT_init.clone()
    eps = eps_init.clone()
    x0 = x0_init.clone()
    log_omega = log_omega_init.clone()
    ux = ux_init.clone()
    K_x = 5
    K_eps = 5
    run_mcmc_and_save(nmcmc, ux, x0, xT, eps, log_omega, device, back_coeff,get_log_omega, tmax=tmax, K_x=K_x, K_eps=K_eps, if_eps=True,path='NWell_MCMC1000/GMM-1000-FP')


