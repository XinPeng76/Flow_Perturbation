
import numpy as np
import torch
import torch.nn as nn
import os
from train_model import ndim,device,Nwellinfo,sig_data
from train_Var import epsilon,tmax
from src.modules.components.common import MLP_nonorm
from utils import EMDSampler, get_energy_device
from src.utils.common import modify_samples_torch_batched_K
from src.modules.EDM import ideal_denoiser
from src.utils.common import generate_tsampling


model = MLP_nonorm(hidden_size= 2000, hidden_layers= 10, emb_size= 80).to(device)  # this is F(x, t)
if os.path.exists(f'models/EMD/{ndim}-d-model.pth'):			
    model.load_state_dict(torch.load(f'models/EMD/{ndim}-d-model.pth'))
else:
    OSError('No model found, please train the model first!')
model.eval()
Sampler = EMDSampler(model, Nwellinfo, sig_data, device)
heun_torch = Sampler.heun_torch

from torch.func import vjp

from functools import partial

 
def score_function_rearange(sig, sig_data, x):
    return (ideal_denoiser(x, sig, sig_data) - x)/sig**2


 
def get_vjp_score(xt, t, sig_data, eps):
    score, vjp_score = vjp(partial(score_function_rearange, t, sig_data), xt)
    x_grad, = vjp_score(eps)
    return score, torch.sum(eps * x_grad, dim=-1)

 
def exact_dynamics_heun_dSt_Huch(tn, tn1, xtn1): # do a lot of Heun steps between tn and tn1 to get the accurate xtn
    ts = generate_tsampling(tn, tn1, 100, 3)
    xt = xtn1 # set the initial x
    dSt = torch.zeros(xt.shape[0]).to(device) # this stores the sum of log(abs(J)) for each step
    
    eps = torch.randn_like(xt)
    _, div_xt = get_vjp_score(xt, ts[len(ts)-1], sig_data, eps)
    #print(score_xt)
    for i in range(len(ts)-1, 0, -1):
        xt_new = heun_torch(ts[i-1], ts[i], xt)
        eps = torch.randn_like(xt_new)
        _, div_xt_new = get_vjp_score(xt_new, ts[i-1], sig_data, eps)

        dSt += (ts[i-1]-ts[i])*(ts[i]*div_xt + ts[i-1]*div_xt_new)/2
        
        div_xt = div_xt_new
        xt = xt_new
    return xt, -dSt

 
def get_log_omega_J(xT):
    x0, deltaSt = exact_dynamics_heun_dSt_Huch(epsilon, tmax, xT) 
    uz = torch.sum(xT**2/tmax**2, dim=-1)/2.0  + 0.5*ndim*np.log(2*np.pi) + ndim*np.log(tmax)
    ux = get_energy_device(x0, Nwellinfo)
    log_omega = -ux + deltaSt + uz

    return log_omega, x0, ux
if __name__ == '__main__':
    if not os.path.exists('NWell_MCMC1000'):
        os.makedirs('NWell_MCMC1000')
    # generate starting point for the MCMC move
    # we need xT_init, log_omega_init
    sampN = 1000
    lambdaT = tmax**2
    xT_init = np.sqrt(lambdaT) * torch.randn(sampN, ndim).to(device)

    log_omega_init, x0_init,ux_init = get_log_omega_J(xT_init)
    K_x = 5
    def save_mcmc_states(xT, log_omega, x0, ux, energy_MC, acceptance_numbers, mc_round, prefix=f'NWell_MCMC1000/GMM-1000-Hutch1-{K_x}'):
        # Create clones of the tensors and move them to CPU
        xT_clone = xT.clone().cpu()
        log_omega_clone = log_omega.clone().cpu()
        x0_clone = x0.clone().cpu()
        ux_clone = ux.clone().cpu()

        # Create a state dictionary
        state_dict = {
            'xT': xT_clone,
            'log_omega': log_omega_clone,
            'x0': x0_clone,
            'ux': ux_clone
        }

        # Save the state dictionary to a file
        torch.save(state_dict, f'{prefix}_{mc_round}.pth')

        # Save other tensors
        torch.save(energy_MC, f'{prefix}_energy_{mc_round}.pt')
        torch.save(acceptance_numbers, f'{prefix}_acceptance_numbers_{mc_round}.pt')

    xT = xT_init.clone()
    x0 = x0_init.clone()
    log_omega = log_omega_init.clone()
    ux = ux_init.clone()
    weights_xT = torch.ones_like(xT, dtype=float).to(device)

     
    nmcmc = 40000 # number of MCMC steps

    log_omega_last_steps = []  # List to store the log_omega of the last 100 steps every 10 steps
    x0_last_steps = []  # List to store the x0 of the last 100 steps every 10 steps
    # 保存每步的能量变化
    energy = []
    acceptance_numbers = []

    #print(xT)
    #print(log_omega)
    for i in range(nmcmc):
        xT_new = xT.clone()
        modify_samples_torch_batched_K(xT_new, weights_xT, mean=0.0, std=tmax,K=K_x)
        #modify_samples_torch_batched(xT_new, mean=0.0, std=tmax)

        log_omega_new, x0_new,ux_new = get_log_omega_J(xT_new)
        #print(log_omega_new)
        db_factor = torch.exp((log_omega_new - log_omega)) # detailed balance move factor
        
        p = torch.rand(db_factor.shape[0]).to(device)

        index_move = p < db_factor # the index to be moved

        xT[index_move] = xT_new[index_move]
        log_omega[index_move] = log_omega_new[index_move]
        x0[index_move] = x0_new[index_move]
        ux[index_move] = ux_new[index_move]
        print(i,ux.mean(),index_move.sum())
        energy.append(ux.mean().cpu())
        acceptance_numbers.append(index_move.sum().cpu())
        #print(xT)
        #print(index_move)
        if i >= 0 and i % 10000 == 0:
            save_mcmc_states(xT, log_omega, x0, ux, energy, acceptance_numbers, i)
            print('sava MCMC:',i)
        if i >= nmcmc - 100 and i % 10 == 0:
            log_omega_last_steps.append(log_omega.cpu())
            x0_last_steps.append(x0.cpu())


    xT = xT.cpu()
    x0 = x0.cpu()
    ux = ux.cpu()
    log_omega = log_omega.cpu()
    # Assuming xT, eps, log_omega, x0, and ux contain your final MCMC states
    state_dict = {
        'xT': xT,
        'log_omega': log_omega,
        'x0': x0,
        'ux': ux
    }

    # Save the state dictionary to a file
    torch.save(state_dict, f'NWell_MCMC1000/GMM-1000-Hutch1-{K_x}.pth')

    concatenated_x0_last_steps = torch.cat(x0_last_steps, dim=0)
    #torch.save(concatenated_x0_last_steps, f'NWell_MCMC1000/x0_last_steps_GMM-1000-one{K_x}.pth')
    torch.save(energy, f'NWell_MCMC1000/energy_GMM-1000-Hutch1-{K_x}.pt')
    torch.save(acceptance_numbers, f'NWell_MCMC1000/acceptance_numbers_GMM-1000-Hutch1-{K_x}.pt')


