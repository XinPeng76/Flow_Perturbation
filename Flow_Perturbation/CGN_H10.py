import torch
import os
import numpy as np
from train_model_CGN import alphas_prod,n_dimensions,n_particles,ndim,device,num_steps,target_energy
from train_Var_CGN import (n_dimensions, n_particles, ndim, back_coeff,time_forward)
from src.utils import remove_mean, clean_up
from utils import load_models_and_data_CGN
from src.MC import get_vjp_score_mnoise, run_mcmc_and_save_CoM

def exact_dynamics_heun_dSt_Huch(xtn1): # do a lot of Heun steps between tn and tn1 to get the accurate xtn
    ts = time_forward
    xt = xtn1 # set the initial x
    dSt = torch.zeros(xt.shape[0]).to(device) # this stores the sum of log(abs(J)) for each step
    nnoise = 10 # we want to average over 10 noises
    eps = torch.randn((nnoise, xt.shape[0], xt.shape[1])).to(device)
    #eps = remove_mean(eps, n_particles, n_dimensions)
    _, div_xt = get_vjp_score_mnoise(xt, ts[len(ts)-1], eps, score_function_rearange)
    for i in range(len(ts)-1, 0, -1):
        xt_new = heun_torch(xt,ts[i], ts[i-1])

        eps = torch.randn((nnoise, xt.shape[0], xt.shape[1])).to(device)
        eps = remove_mean(eps, n_particles, n_dimensions)
        _, div_xt_new = get_vjp_score_mnoise(xt_new, ts[i-1], eps, score_function_rearange)

        dSt += (ts[i]-ts[i-1])*(div_xt + div_xt_new)/2
        
        div_xt = div_xt_new
        xt = xt_new
        xt = remove_mean(xt, n_particles, n_dimensions)
    return xt, -dSt

 
def get_log_omega_J(xT, eps=None):
    x0, deltaSt = exact_dynamics_heun_dSt_Huch(xT) 
    uz = torch.sum(xT**2, dim=-1)/2.0  + 0.5*ndim*np.log(2*np.pi)
    ux = target_energy.energy(x0).squeeze(1)
    log_omega = -ux + deltaSt + uz

    return log_omega, x0, ux

if __name__ == '__main__':
    if not os.path.exists('data'):
        os.makedirs('data')
    heun_torch, score_function_rearange,score_function_1element = load_models_and_data_CGN(ndim, num_steps, alphas_prod, n_particles, n_dimensions, back_coeff)
    sampN = 10000
    nmcmc = 60000

    xT_init = torch.randn(sampN, ndim).to(device)
    xT_init = remove_mean(xT_init, n_particles, n_dimensions)

    log_omega_init, x0_init, ux_init = get_log_omega_J(xT_init)
    print(ux_init.min())
    #print(x0_init.shape)

    samples_selected = ux_init < 2000

    xT = xT_init[samples_selected].clone()
    x0 = x0_init[samples_selected].clone()
    log_omega = log_omega_init[samples_selected].clone()
    ux = ux_init[samples_selected].clone()
    '''

    data = torch.load('mcmc_final_state_one1.pth')

    xT = data['xT'].to(device)
    x0 = data['x0'].to(device)
    ux = data['ux'].to(device)
    log_omega = data['log_omega'].to(device)
    '''
    eps = torch.randn_like(xT)
    print(xT.shape)
    run_mcmc_and_save_CoM(nmcmc, ux, x0, xT, eps, log_omega, n_particles, n_dimensions, device, back_coeff,get_log_omega_J, K_x=1, K_eps=1, if_eps=False,path='data/CGN-H10')
    clean_up()


