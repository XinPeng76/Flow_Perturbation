import torch
import os
import numpy as np
from train_model_CGN import alphas_prod,n_dimensions,n_particles,ndim,device,num_steps,target_energy
from train_Var_CGN import (n_dimensions, n_particles, ndim, back_coeff,time_forward)
from src.utils import remove_mean, clean_up
from utils import load_models_and_data_CGN
from src.MC import get_jacobian_score, run_mcmc_and_save_CoM
def exact_dynamics_heun_dSt(xtn1): # do a lot of Heun steps between tn and tn1 to get the accurate xtn
    ts = time_forward
    xt = remove_mean(xtn1, n_particles, n_dimensions) # set the initial x
    dSt = torch.zeros(xt.shape[0]).to(device) # this stores the sum of log(abs(J)) for each step
    
    jj_score_xt = v_jacobian_score(xt, ts[len(ts)-1]).to(device)
    div_xt = torch.einsum("...ii", jj_score_xt)
    for i in range(len(ts)-1, 0, -1):
        xt_new = heun_torch(xt,ts[i], ts[i-1])
        jj_score_xt_new = v_jacobian_score(xt_new, ts[i-1]).to(device)
        div_xt_new = torch.einsum("...ii", jj_score_xt_new)
        dSt += (ts[i]-ts[i-1])*(div_xt + div_xt_new)/2
        div_xt = div_xt_new
        xt = xt_new
        xt = remove_mean(xt, n_particles, n_dimensions)
        del xt_new,jj_score_xt_new,div_xt_new
    return xt, -dSt

def get_log_omega_J(xT, eps=None):
    x0, deltaSt = exact_dynamics_heun_dSt(xT) 
    #print(x0.shape)
    uz = torch.sum(xT**2, dim=-1)/2.0  + 0.5*ndim*np.log(2*np.pi)
    ux = target_energy.energy(x0).squeeze(1)
    log_omega = -ux + deltaSt + uz

    return log_omega, x0, ux
if __name__ == '__main__':
    if not os.path.exists('data'):
        os.makedirs('data') 
    # generate starting point for the MCMC move
    # we need xT_init, eps_init, log_omega_init
    # do the backward-forward dynamics, get deltaSt, determine forward coeff
    K_x = 1
    heun_torch, score_function_rearange,score_function_1element = load_models_and_data_CGN(ndim, num_steps, alphas_prod, n_particles, n_dimensions, back_coeff)
    v_jacobian_score = get_jacobian_score(score_function_1element)
    def process_batch(batch_data, n_particles, n_dimensions):
        #batch_data = remove_mean(batch_data, n_particles, n_dimensions)
        log_omega_batch, x0_batch, ux_batch = get_log_omega_J(batch_data)
        samples_selected = ux_batch < 2000
        return (
            batch_data[samples_selected].clone(),
            x0_batch[samples_selected].clone(),
            log_omega_batch[samples_selected].clone(),
            ux_batch[samples_selected].clone()
        )
    sampN = 10000
    nmcmc = 4000
    batch_size = 750
    xT_init = torch.randn(sampN, ndim).to(device)
    xT_init = remove_mean(xT_init, n_particles, n_dimensions)

    #log_omega_init, x0_init, ux_init = get_log_omega_J(xT_init)
    #print(ux_init.min())
    #print(x0_init.shape)

    #samples_selected = ux_init < 2000

    #xT = xT_init[samples_selected].clone()
    #x0 = x0_init[samples_selected].clone()
    #log_omega = log_omega_init[samples_selected].clone()
    #ux = ux_init[samples_selected].clone()
    
    xT_list, x0_list, log_omega_list, ux_list = [], [], [], []

    for i in range(0, sampN, batch_size):
        batch_data = xT_init[i:i + batch_size if i + batch_size < sampN else sampN]
        xT_batch, x0_batch, log_omega_batch, ux_batch = process_batch(batch_data, n_particles, n_dimensions)
        
        xT_list.append(xT_batch)
        x0_list.append(x0_batch)
        log_omega_list.append(log_omega_batch)
        ux_list.append(ux_batch)

    xT = torch.cat(xT_list, dim=0).to(device)
    x0 = torch.cat(x0_list, dim=0).to(device)
    log_omega = torch.cat(log_omega_list, dim=0).to(device)
    ux = torch.cat(ux_list, dim=0).to(device)
    '''
    data = torch.load(f'CGN/CGN-J-{back_coeff}-{K_x}_{4000}.pth')

    xT = data['xT'].to(device)
    x0 = data['x0'].to(device)
    ux = data['ux'].to(device)
    log_omega = data['log_omega'].to(device)
    '''   
    eps = torch.randn_like(xT)
    print(xT.shape)
    run_mcmc_and_save_CoM(nmcmc, ux, x0, xT, eps, log_omega, n_particles, n_dimensions, device, back_coeff, get_log_omega_J, K_x, K_eps=1, if_eps=False, path='data/CGN-J')
    clean_up()

     



