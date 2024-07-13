import torch
import os
from bgmol.datasets import ChignolinOBC2PT
import numpy as np
from train_model_CGN import alphas_prod,n_dimensions,n_particles,ndim,device,num_steps
from src.DDPM import interpolate_parameters
from utils import DDPMSamplerCOM
from src.common import MLP
from train_Var_CGN import (n_dimensions, n_particles, ndim, back_coeff,time_forward)
from src.utils import remove_mean, modify_samples_torch_batched_K,clean_up

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

is_data_here = os.path.isdir("ChignolinOBC2PT")
dataset = ChignolinOBC2PT(download=not is_data_here, read=True)

#system = dataset.system

model = MLP(ndim=ndim,hidden_size=2048,hidden_layers=12).to(device)		
if os.path.exists('models/CGN/model_RotAug_LowLR.pth'):			
    model.load_state_dict(torch.load('models/CGN/model_RotAug_LowLR.pth'))
else:
    OSError('No model found, please train the model first!')
model.eval()

st, sigma_t, st_derivative, sigma_t_derivative = interpolate_parameters(num_steps, alphas_prod)
Sampler = DDPMSamplerCOM(model, st, st_derivative, sigma_t_derivative, n_particles, n_dimensions, device)

heun_torch = Sampler.heun_torch
target_energy = dataset.get_energy_model(n_simulation_steps=0)


for param in model.parameters():
    param.requires_grad = False

def score_function_rearange(t, x):
    t_repeat = (t*torch.ones(x.shape[0])).to(device)
    pred_noise = model(x, t_repeat)
    pred_noise = remove_mean(pred_noise, n_particles, n_dimensions)
    return st_derivative(t)/st(t) * x + st(t) * sigma_t_derivative(t) * pred_noise

def score_function(x,t):
    t_repeat = (t*torch.ones(x.shape[0])).to(device)
    pred_noise = model(x, t_repeat)
    pred_noise = remove_mean(pred_noise, n_particles, n_dimensions)
    return st_derivative(t)/st(t) * x + st(t) * sigma_t_derivative(t) * pred_noise

def score_function_1element(x, t):
    #print(x)
    x = x.reshape(1,-1)
    t_repeat = (t*torch.ones(x.shape[0])).to(device)
    pred_noise = model(x, t_repeat)
    pred_noise = remove_mean(pred_noise, n_particles, n_dimensions)
    score = st_derivative(t)/st(t) * x + st(t) * sigma_t_derivative(t) * pred_noise
    return score.reshape(-1)

from torch.func import jacrev, vmap

jacobian_score = jacrev(score_function_1element)
v_jacobian_score = vmap(jacobian_score, in_dims=(0, None))

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
        dSt += (ts[i-1]-ts[i])*(div_xt + div_xt_new)/2
        div_xt = div_xt_new
        xt = xt_new
        xt = remove_mean(xt, n_particles, n_dimensions)
        del xt_new,jj_score_xt_new,div_xt_new
    return xt, -dSt

def get_log_omega_J(xT):
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
    batch_size = 200
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
    def save_mcmc_states(xT, log_omega, x0, ux, energy_MC_ten, acceptance_numbers, mc_round, prefix=f'data/CGN-J-{back_coeff}-{K_x}'):
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
        torch.save(energy_MC_ten, f'{prefix}_energy_{mc_round}.pt')
        torch.save(acceptance_numbers, f'{prefix}_acceptance_numbers_{mc_round}.pt')


    print(xT.shape)
     

     
    nmcmc = 4500 # number of MCMC steps

    log_omega_last_steps = []  # List to store the log_omega of the last 100 steps every 10 steps
    x0_last_steps = []  # List to store the x0 of the last 100 steps every 10 steps
    acceptance_numbers = []
    energy = []
    #energy = torch.load(f'CGN/CGN-J-{back_coeff}-{K_x}_energy_{4000}.pt')
    #energy = list(energy)
    weights_xT = torch.ones_like(xT, dtype=float).to(device)

    for i in range(0,nmcmc):
        xT_new = xT.clone()

        modify_samples_torch_batched_K(xT_new, weights_xT, mean=0.0, std=1.0,K=K_x)
        
        xT_new = remove_mean(xT_new, n_particles, n_dimensions)
        
        log_omega_new, x0_new, ux_new = get_log_omega_J(xT_new)
        #print(log_omega)
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
        #print(log_omega)
        #print(index_move)
        if i >= 0 and i % 500 == 0:
            save_mcmc_states(xT, log_omega, x0, ux, energy, acceptance_numbers, i)
        if i >= nmcmc - 100 and i % 10 == 0:
            log_omega_last_steps.append(log_omega.cpu())
            x0_last_steps.append(x0.cpu())

     
    # Assuming xT, eps, log_omega, x0, and ux contain your final MCMC states
    state_dict = {
        'xT': xT,
        'log_omega': log_omega,
        'x0': x0,
        'ux': ux
    }

    # Save the state dictionary to a file
    torch.save(state_dict, f'data/CGN-J-{back_coeff}-{K_x}.pth')

    concatenated_x0_last_steps = torch.cat(x0_last_steps, dim=0)
    #torch.save(concatenated_x0_last_steps, f'CGN/x0_last_steps_CGN-J-{back_coeff}-{K_x}.pth')
    torch.save(energy, f'data/energy_CGN-J-{back_coeff}-{K_x}.pt')
    torch.save(acceptance_numbers, f'data/acceptance_numbers_CGN-J-{back_coeff}-{K_x}.pt')
    clean_up()

     



