import torch
import os
from bgmol.datasets import ChignolinOBC2PT
import numpy as np
from train_model import alphas_prod,n_dimensions,n_particles,ndim,device,num_steps
from src.modules.DDPM import interpolate_parameters
from utils import DDPMSamplerCOM
from src.modules.components.common import MLP,  MLP_var
from train_Var import (n_dimensions, n_particles, ndim, back_coeff,time_forward, time_backward)
from src.utils.common import remove_mean, modify_samples_torch_batched_K

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

is_data_here = os.path.isdir("ChignolinOBC2PT")
dataset = ChignolinOBC2PT(download=not is_data_here, read=True)

system = dataset.system

model = MLP(hidden_size=2048,hidden_layers=12).to(device)			# 输出维度是2，输入是x和step
if os.path.exists('models/CGN/model_RotAug_LowLR.pth'):			
    model.load_state_dict(torch.load('models/CGN/model_RotAug_LowLR.pth'))
else:
    OSError('No model found, please train the model first!')
model.eval()

st, sigma_t, st_derivative, sigma_t_derivative = interpolate_parameters(num_steps, alphas_prod)
Sampler = DDPMSamplerCOM(model, st, st_derivative, sigma_t_derivative, n_particles, n_dimensions, device)

ddpm_ode_heun = Sampler.exact_dynamics_heun
target_energy = dataset.get_energy_model(n_simulation_steps=0)

model_var = MLP_var().to(device)
if os.path.exists('models/CGN/model_RotAug_LowLR_Var_{}.pth'.format(back_coeff)):
    model_var.load_state_dict(torch.load('models/CGN/model_RotAug_LowLR_Var_{}.pth'.format(back_coeff)))
else:
    OSError('No model_var_{} found, please train the model_var first!'.format(back_coeff))
model_var.eval()

for param in model.parameters():
    param.requires_grad = False

for param in model_var.parameters():
    param.requires_grad = False

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
    sampN = 10000
    xT_init = torch.randn(sampN, ndim).to(device)
    xT_init = remove_mean(xT_init, n_particles, n_dimensions)
    eps_init = torch.randn_like(xT_init)
    eps_init = remove_mean(eps_init, n_particles, n_dimensions)

    log_omega_init, x0_init, ux_init = get_log_omega(xT_init, eps_init)
    samples_selected = ux_init < 20000
    xT = xT_init[samples_selected].clone()
    eps = eps_init[samples_selected].clone()
    x0 = x0_init[samples_selected].clone()
    log_omega = log_omega_init[samples_selected].clone()
    ux = ux_init[samples_selected].clone()

    nmcmc = 20000 # number of MCMC steps

    log_omega_last_steps = []  # List to store the log_omega of the last 100 steps every 10 steps
    x0_last_steps = []  # List to store the x0 of the last 100 steps every 10 steps
    weights_xT = torch.ones_like(xT, dtype=float).to(device)
    weights_eps = torch.ones_like(eps, dtype=float).to(device)
    energy = []
    K_x = 5
    K_eps = 5
    #print(xT)
    #print(log_omega)
    for i in range(nmcmc):
        xT_new = xT.clone()
        eps_new = eps.clone()

        modify_samples_torch_batched_K(xT_new, weights_xT, mean=0.0, std=1.0, K=K_x)
        modify_samples_torch_batched_K(eps_new, weights_eps, mean=0.0, std=1.0, K=K_eps)
        
        xT_new = remove_mean(xT_new, n_particles, n_dimensions)
        eps_new = remove_mean(eps_new, n_particles, n_dimensions)
        
        log_omega_new, x0_new, ux_new = get_log_omega(xT_new, eps_new)
        #print(log_omega)
        db_factor = torch.exp((log_omega_new - log_omega)) # detailed balance move factor
        
        p = torch.rand(db_factor.shape[0]).to(device)

        index_move = p < db_factor # the index to be moved
        #print(db_factor.shape)
        #print(index_move.shape)

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
    eps = eps.cpu()
    x0 = x0.cpu()
    ux = ux.cpu()
    # Assuming xT, eps, log_omega, x0, and ux contain your final MCMC states
    state_dict = {
        'xT': xT,
        'eps': eps,
        'log_omega': log_omega,
        'x0': x0,
        'ux': ux
    }

        # Save the state dictionary to a file
    torch.save(state_dict, f'data/CGN-FP-{back_coeff}-{K_x}-{K_eps}.pth')

    concatenated_x0_last_steps = torch.cat(x0_last_steps, dim=0)
    torch.save(concatenated_x0_last_steps, f'data/x0_last_steps_CGN-FP-{back_coeff}-{K_x}-{K_eps}.pth')
    torch.save(energy, f'data/energy_CGN-FP-{back_coeff}-{K_x}-{K_eps}.pt')



