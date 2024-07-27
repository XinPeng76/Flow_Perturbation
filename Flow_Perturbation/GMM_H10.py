
import numpy as np
import torch
import os
from train_model_GMM import ndim,device,Nwellinfo,sig_data
from train_Var_GMM import back_coeff,epsilon,tmax
from src.GMM_distribution import get_energy_device
from utils import load_models_and_data_GMM
from src.utils import generate_tsampling
from src.MC import run_mcmc_and_save, get_vjp_score_mnoise

def exact_dynamics_heun_dSt_Huch(tn, tn1, xtn1): # do a lot of Heun steps between tn and tn1 to get the accurate xtn
    ts = generate_tsampling(tn, tn1, 100, 3)
    xt = xtn1 # set the initial x
    dSt = torch.zeros(xt.shape[0]).to(device) # this stores the sum of log(abs(J)) for each step
    nnoise = 10 # we want to average over 10 noises
    eps = torch.randn((nnoise, xt.shape[0], xt.shape[1])).to(device)
    _, div_xt = get_vjp_score_mnoise(xt, ts[len(ts)-1], eps, score_function_rearange)
    for i in range(len(ts)-1, 0, -1):
        xt_new = heun_torch(ts[i-1], ts[i], xt)

        eps = torch.randn((nnoise, xt.shape[0], xt.shape[1])).to(device)
        _, div_xt_new = get_vjp_score_mnoise(xt_new, ts[i-1], eps, score_function_rearange)

        dSt += (ts[i-1]-ts[i])*(ts[i]*div_xt + ts[i-1]*div_xt_new)/2
        
        div_xt = div_xt_new
        xt = xt_new
    return xt, -dSt

def get_log_omega_J(xT, eps=None):
    x0, deltaSt = exact_dynamics_heun_dSt_Huch(epsilon, tmax, xT) 
    uz = torch.sum(xT**2/tmax**2, dim=-1)/2.0  + 0.5*ndim*np.log(2*np.pi) + ndim*np.log(tmax)
    ux = get_energy_device(x0, Nwellinfo)
    log_omega = -ux + deltaSt + uz

    return log_omega, x0, ux

if __name__ == '__main__':
    if not os.path.exists('NWell_MCMC1000'):
        os.makedirs('NWell_MCMC1000')
    heun_torch, score_function_rearange, score_function_1element = load_models_and_data_GMM(ndim, device, back_coeff, sig_data, load_var=False)
    # generate starting point for the MCMC move
    # we need xT_init, log_omega_init
    sampN = 1000
    nmcmc = 40000
    lambdaT = tmax**2
    xT_init = np.sqrt(lambdaT) * torch.randn(sampN, ndim).to(device)

    log_omega_init, x0_init,ux_init = get_log_omega_J(xT_init)
    K_x = 5
    xT = xT_init.clone()
    x0 = x0_init.clone()
    log_omega = log_omega_init.clone()
    ux = ux_init.clone()
    eps = torch.randn_like(xT)
    run_mcmc_and_save(nmcmc, ux, x0, xT, eps, log_omega, device, back_coeff,get_log_omega_J, tmax=tmax, K_x=K_x, if_eps=False,path='NWell_MCMC1000/GMM-1000-Hutch10')



