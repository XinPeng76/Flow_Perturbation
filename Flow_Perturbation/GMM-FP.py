
import numpy as np
import torch
import torch.nn as nn
import os
from train_model_GMM import ndim,device,Nwellinfo,sig_data
from train_Var_GMM import back_coeff,epsilon,tmax
from src.common import MLP_nonorm,  MLP_var
from utils import EMDSampler, get_energy_device
from src.utils import modify_samples_torch_batched_K

model = MLP_nonorm(ndim=ndim,hidden_size= 2000, hidden_layers= 10, emb_size= 80).to(device)  # this is F(x, t)
if os.path.exists(f'models/GMM/{ndim}-d-model.pth'):			
    model.load_state_dict(torch.load(f'models/GMM/{ndim}-d-model.pth'))
else:
    OSError('No model found, please train the model first!')
model.eval()
for param in model.parameters():
    param.requires_grad = False
# Create the model
model_var = MLP_var(ndim=ndim).to(device)
model_var.load_state_dict(torch.load('models/GMM/model_var_{}.pt'.format(back_coeff)))
model_var.eval()
for param in model_var.parameters():
    param.requires_grad = False

Sampler = EMDSampler(model, sig_data, device)

exact_dynamics_heun = Sampler.exact_dynamics_heun


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

    # generate starting point for the MCMC move
    # we need xT_init, eps_init, log_omega_init
    # do the backward-forward dynamics, get deltaSt, determine forward coeff
    sampN = 1000
    lambdaT = tmax**2
    xT_init = np.sqrt(lambdaT) * torch.randn(sampN, ndim).to(device)
    eps_init = torch.randn_like(xT_init)

    log_omega_init, x0_init,ux_init = get_log_omega(xT_init, eps_init)

    xT = xT_init.clone()
    eps = eps_init.clone()
    x0 = x0_init.clone()
    log_omega = log_omega_init.clone()
    ux = ux_init.clone()

    nmcmc = 4000 # number of MCMC steps

    log_omega_last_steps = []  # List to store the log_omega of the last 100 steps every 10 steps
    x0_last_steps = []  # List to store the x0 of the last 100 steps every 10 steps
    weights_xT = torch.ones_like(xT, dtype=float).to(device)
    weights_eps = torch.ones_like(eps, dtype=float).to(device)
    energy = []
    #print(xT)
    #print(log_omega)
    K_x = 5
    K_eps = 5
    for i in range(nmcmc):
        xT_new = xT.clone()
        eps_new = eps.clone()

        #p_mut = torch.rand(1).item()
        #if (p_mut < 0.5):
        #    modify_samples_torch_batched(xT_new, mean=0.0, std=tmax)
        #else:
        #    modify_samples_torch_batched(eps_new, mean=0.0, std=1.0)
        modify_samples_torch_batched_K(xT_new, weights_xT, mean=0.0, std=tmax, K=K_x )
        modify_samples_torch_batched_K(eps_new, weights_eps, mean=0.0, std=1.0, K=K_eps)

        log_omega_new, x0_new, ux_new = get_log_omega(xT_new, eps_new)
        #print(log_omega_new)
        db_factor = torch.exp((log_omega_new - log_omega)) # detailed balance move factor
        
        p = torch.rand(db_factor.shape[0]).to(device)

        index_move = p < db_factor # the index to be moved

        xT[index_move] = xT_new[index_move]
        eps[index_move] = eps_new[index_move]
        log_omega[index_move] = log_omega_new[index_move]
        x0[index_move] = x0_new[index_move]
        ux[index_move] = ux_new[index_move]
        print(i,ux.mean(),index_move.sum())
        energy.append(ux.mean().cpu())
        #print(xT)
        #print(index_move)
        #print(log_omega)
        if i >= nmcmc - 100 and i % 10 == 0:
            log_omega_last_steps.append(log_omega.cpu())
            x0_last_steps.append(x0.cpu())
        #print(i, index_move.sum())

    xT = xT.cpu()
    x0 = x0.cpu()
    ux = ux.cpu()
    log_omega = log_omega.cpu()
    eps = eps.cpu()
    # Assuming xT, eps, log_omega, x0, and ux contain your final MCMC states
    state_dict = {
        'xT': xT,
        'log_omega': log_omega,
        'x0': x0,
        'ux': ux,
        'eps':eps
    }

    # Save the state dictionary to a file
    torch.save(state_dict, f'NWell_MCMC1000/GMM-1000-u-{back_coeff}-{K_x}-{K_eps}.pth')

    concatenated_x0_last_steps = torch.cat(x0_last_steps, dim=0)
    torch.save(concatenated_x0_last_steps, f'NWell_MCMC1000/x0_last_steps_GMM-1000-u-{back_coeff}-{K_x}-{K_eps}.pth')
    torch.save(energy, f'NWell_MCMC1000/energy_GMM-1000-u-{back_coeff}-{K_x}-{K_eps}.pt')


