import numpy as np
import torch
import torch.nn as nn
import os
from train_model_GMM import ndim,device,Nwellinfo,sig_data
from train_Var_GMM import epsilon,tmax
from src.common import MLP_nonorm
from utils import EMDSampler, get_energy_device
from src.utils import modify_samples_torch_batched_K
from src.utils import generate_tsampling


model = MLP_nonorm(ndim=ndim,hidden_size= 2000, hidden_layers= 10, emb_size= 80).to(device)  # this is F(x, t)
if os.path.exists(f'models/GMM/{ndim}-d-model.pth'):			
    model.load_state_dict(torch.load(f'models/GMM/{ndim}-d-model.pth'))
else:
    OSError('No model found, please train the model first!')
model.eval()
for param in model.parameters():
    param.requires_grad = False
Sampler = EMDSampler(model, sig_data, device)
heun_torch = Sampler.heun_torch
ideal_denoiser = Sampler.ideal_denoiser

 
def score_function_rearange(sig, sig_data, x):
    return (ideal_denoiser(x, sig, sig_data) - x)/sig**2


 
# now do the Jocobian based deltaSt calculation 
def score_function_1element(x, sig, sig_data): # x is of shape (ndim,)
    x = x.reshape(1,-1)
    score = (ideal_denoiser(x, sig, sig_data) - x)/sig**2
    return score.flatten()

 
from torch.func import jacrev, vmap

 
def score_function_rearange(sig, sig_data, x):
    return (ideal_denoiser(x, sig, sig_data) - x)/sig**2

 
jacobian_score = jacrev(score_function_1element)
v_jacobian_score = vmap(jacobian_score, in_dims=(0, None, None))

 
def v_jacobian_score_batch(x, t, sig_data, batch_size=200):
    nbatch = x.size(0)
    n = nbatch // batch_size
    scores = []
    for i in range(n):
        x_batch = x[i*batch_size:(i+1)*batch_size]
        scores.append(v_jacobian_score(x_batch, t, sig_data))
    if nbatch % batch_size > 0:
        x_batch = x[n*batch_size:]
        scores.append(v_jacobian_score(x_batch, t, sig_data))
    return torch.cat(scores, dim=0)


def exact_dynamics_heun_dSt(tn, tn1, xtn1): # do a lot of Heun steps between tn and tn1 to get the accurate xtn
    ts = generate_tsampling(tn, tn1, 100, 3)
    xt = xtn1 # set the initial x
    dSt = torch.zeros(xt.shape[0]).to(device) # this stores the sum of log(abs(J)) for each step
    
    jj_score_xt = v_jacobian_score_batch(xt, ts[len(ts)-1], sig_data)
    div_xt = torch.einsum("...ii", jj_score_xt)
    for i in range(len(ts)-1, 0, -1):
        xt_new = heun_torch(ts[i-1], ts[i], xt)
        jj_score_xt_new = v_jacobian_score_batch(xt_new, ts[i-1], sig_data)
        div_xt_new = torch.einsum("...ii", jj_score_xt_new)
        dSt += (ts[i-1]-ts[i])*(ts[i]*div_xt + ts[i-1]*div_xt_new)/2
        div_xt = div_xt_new
        xt = xt_new
        del xt_new,jj_score_xt_new,div_xt_new
    return xt, -dSt

 
def exact_dynamics_heun_dSt(tn, tn1, xtn1): # do a lot of Heun steps between tn and tn1 to get the accurate xtn
    ts = generate_tsampling(tn, tn1, 100, 3)
    xt = xtn1 # set the initial x
    dSt = torch.zeros(xt.shape[0]).to(device) # this stores the sum of log(abs(J)) for each step
    
    jj_score_xt = v_jacobian_score_batch(xt, ts[len(ts)-1], sig_data)
    div_xt = torch.einsum("...ii", jj_score_xt)
    for i in range(len(ts)-1, 0, -1):
        xt_new = heun_torch(ts[i-1], ts[i], xt)
        jj_score_xt_new = v_jacobian_score_batch(xt_new, ts[i-1], sig_data)
        div_xt_new = torch.einsum("...ii", jj_score_xt_new)
        dSt += (ts[i-1]-ts[i])*(ts[i]*div_xt + ts[i-1]*div_xt_new)/2
        div_xt = div_xt_new
        xt = xt_new
    return xt, -dSt

 
def get_log_omega_J(xT):
    x0, deltaSt = exact_dynamics_heun_dSt(epsilon, tmax, xT) 
    uz = torch.sum(xT**2/tmax**2, dim=-1)/2.0  + 0.5*ndim*np.log(2*np.pi) + ndim*np.log(tmax)
    ux = get_energy_device(x0, Nwellinfo)
    log_omega = -ux + deltaSt + uz

    return log_omega, x0, ux

if __name__ == '__main__':
    if not os.path.exists('NWell_MCMC1000'):
        os.makedirs('NWell_MCMC1000')
    sampN = 1000
    lambdaT = tmax**2
    xT_init = np.sqrt(lambdaT) * torch.randn(sampN, ndim).to(device)
    
    log_omega_init, x0_init, ux_init = get_log_omega_J(xT_init)

    
    xT = xT_init.clone()
    x0 = x0_init.clone()
    log_omega = log_omega_init.clone()
    ux = ux_init.clone()
    weights_xT = torch.ones_like(xT, dtype=float).to(device)

    del log_omega_init, xT_init, x0_init, ux_init
    torch.cuda.empty_cache()
    K_x = 5
    def save_mcmc_states(xT, log_omega, x0, ux, energy_MC_ten, acceptance_numbers, mc_round, prefix=f'NWell_MCMC1000/GMM-1000-J-{K_x}'):
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
    
    nmcmc = 4000 # number of MCMC steps

    log_omega_last_steps = []  # List to store the log_omega of the last 100 steps every 10 steps
    x0_last_steps = []  # List to store the x0 of the last 100 steps every 10 steps
    energy_J = []
    acceptance_numbers = []

    #print(xT)
    #print(log_omega)
    for i in range(nmcmc):
        xT_new = xT.clone()

        #modify_samples_torch_batched(xT_new, mean=0.0, std=tmax)
        modify_samples_torch_batched_K(xT_new, weights_xT, mean=0.0, std=tmax,K=K_x)

        log_omega_new, x0_new, ux_new = get_log_omega_J(xT_new)
        #print(log_omega_new)
        db_factor = torch.exp((log_omega_new - log_omega)) # detailed balance move factor
        
        p = torch.rand(db_factor.shape[0]).to(device)

        index_move = p < db_factor # the index to be moved

        xT[index_move] = xT_new[index_move]
        log_omega[index_move] = log_omega_new[index_move]
        x0[index_move] = x0_new[index_move]
        ux[index_move] = ux_new[index_move]
        print(i,ux.mean(),index_move.sum())
        acceptance_numbers.append(index_move.sum().cpu())
        energy_J.append(ux.mean())
        if i >= 0 and i % 500 == 0:
            save_mcmc_states(xT, log_omega, x0, ux, energy_J, acceptance_numbers, i)
        if i >= nmcmc - 100 and i % 10 == 0:
            log_omega_last_steps.append(log_omega.cpu())
            x0_last_steps.append(x0.cpu())
        #print(i, index_move.sum())


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
    torch.save(state_dict, f'NWell_MCMC1000/GMM-1000-J-{K_x}.pth')

    concatenated_x0_last_steps = torch.cat(x0_last_steps, dim=0)
    torch.save(concatenated_x0_last_steps, f'NWell_MCMC1000/x0_last_steps_GMM-1000-J-{K_x}.pth')
    torch.save(energy_J, f'NWell_MCMC1000/energy_GMM-1000-J-{K_x}.pt')


