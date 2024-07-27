import torch
import numpy as np

from torch.func import vjp, vmap, jacrev
from functools import partial
from .utils import remove_mean, modify_samples_torch_batched_K

def get_vjp_score(xt, t, eps, score_function_rearange):
    score, vjp_score = vjp(partial(score_function_rearange, t), xt)
    x_grad, = vjp_score(eps)
    return score, torch.sum(eps * x_grad, dim=-1)

def get_vjp_score_mnoise(xt, t, eps, score_function_rearange):
    score, vjp_score = vjp(partial(score_function_rearange, t), xt)
    x_grad, = vmap(vjp_score)(eps)
    return score, torch.mean(torch.sum(eps * x_grad, dim=-1),dim=0)

def get_jacobian_score(score_function_1element):
    jacobian_score = jacrev(score_function_1element)
    v_jacobian_score = vmap(jacobian_score, in_dims=(0, None))
    return v_jacobian_score

def v_jacobian_score_batch(x, t, v_jacobian_score, batch_size=200):
    nbatch = x.size(0)
    n = nbatch // batch_size
    scores = []
    for i in range(n):
        x_batch = x[i*batch_size:(i+1)*batch_size]
        scores.append(v_jacobian_score(x_batch, t))
    if nbatch % batch_size > 0:
        x_batch = x[n*batch_size:]
        scores.append(v_jacobian_score(x_batch, t))
    return torch.cat(scores, dim=0)

def save_mcmc_states(xT, log_omega, x0, ux, energy, acceptance_numbers, mc_round, path):
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
    torch.save(state_dict, f'{path}_{mc_round}.pth')

    # Save other tensors
    torch.save(energy, f'{path}_energy_{mc_round}.pt')
    torch.save(acceptance_numbers, f'{path}_acceptance_numbers_{mc_round}.pt')

def run_mcmc_and_save_CoM(nmcmc, ux, x0, xT, eps, log_omega, n_particles, n_dimensions, device, back_coeff,get_log_omega, tmax=1.0,
                            K_x=1, K_eps=1, if_sava=True,  if_eps=False,path='data/CGN-FP',last_steps=100,count_last=10,count_sava=10):
    energy = []
    log_omega_last_steps = []
    x0_last_steps = []
    acceptance_numbers = []
    weights_xT = torch.ones_like(xT, dtype=float).to(device)
    weights_eps = torch.ones_like(eps, dtype=float).to(device)
    for i in range(nmcmc):
        xT_new = xT.clone()
        eps_new = eps.clone()

        modify_samples_torch_batched_K(xT_new, weights_xT, mean=0.0, std=tmax, K=K_x)
        if if_eps:
            modify_samples_torch_batched_K(eps_new, weights_eps, mean=0.0, std=1.0, K=K_eps)
            eps_new = remove_mean(eps_new, n_particles, n_dimensions)
        xT_new = remove_mean(xT_new, n_particles, n_dimensions)
        log_omega_new, x0_new, ux_new = get_log_omega(xT_new, eps_new)
        db_factor = torch.exp((log_omega_new - log_omega))  # detailed balance move factor
        
        p = torch.rand(db_factor.shape[0]).to(device)
        index_move = p < db_factor  # the index to be moved
        #print(xT.shape,x0.shape,index_move.shape,db_factor.shape)

        xT[index_move] = xT_new[index_move]
        eps[index_move] = eps_new[index_move]
        log_omega[index_move] = log_omega_new[index_move]
        x0[index_move] = x0_new[index_move]
        ux[index_move] = ux_new[index_move]
        print(i, ux.mean(), index_move.sum())
        energy.append(ux.mean().cpu())
        acceptance_numbers.append(index_move.sum().cpu())
        if  i >= 0 and i % (nmcmc//count_sava) == 0 and if_sava:
            save_mcmc_states(xT, log_omega, x0, ux, energy, acceptance_numbers, i, f'{path}-{back_coeff}-{K_x}-{K_eps}')
        if i >= nmcmc - last_steps and i % (last_steps//count_last) == 0:
            log_omega_last_steps.append(log_omega.cpu())
            x0_last_steps.append(x0.cpu())

    xT = xT.cpu()
    eps = eps.cpu()
    x0 = x0.cpu()
    ux = ux.cpu()
    state_dict = {
        'xT': xT,
        'eps': eps,
        'log_omega': log_omega,
        'x0': x0,
        'ux': ux
    }

    # Save the state dictionary to a file
    #torch.save(state_dict, f'data/CGN-FP-{back_coeff}-{K_x}-{K_eps}.pth')
    concatenated_x0_last_steps = torch.cat(x0_last_steps, dim=0)
    #torch.save(concatenated_x0_last_steps, f'data/x0_last_steps_CGN-FP-{back_coeff}-{K_x}-{K_eps}.pth')
    #torch.save(energy, f'data/energy_CGN-FP-{back_coeff}-{K_x}-{K_eps}.pt')
    torch.save(state_dict, f'{path}-{back_coeff}-{K_x}-{K_eps}.pth')
    torch.save(concatenated_x0_last_steps, f'{path}-x0_last_steps-{back_coeff}-{K_x}-{K_eps}.pth')
    torch.save(energy, f'{path}-energy-{back_coeff}-{K_x}-{K_eps}.pt')
    return concatenated_x0_last_steps

def run_mcmc_and_save(nmcmc, ux, x0, xT, eps, log_omega, device, back_coeff,get_log_omega, 
                        tmax=1.0, K_x=5, K_eps=5, if_sava=True, if_eps=False,path='GMM/GMM-FP',last_steps=100,count_last=10,count_sava=10):
    energy = []
    log_omega_last_steps = []
    x0_last_steps = []
    acceptance_numbers = []
    weights_xT = torch.ones_like(xT, dtype=float).to(device)
    weights_eps = torch.ones_like(eps, dtype=float).to(device)
    for i in range(nmcmc):
        xT_new = xT.clone()
        eps_new = eps.clone()

        modify_samples_torch_batched_K(xT_new, weights_xT, mean=0.0, std=tmax, K=K_x)
        if if_eps:
            modify_samples_torch_batched_K(eps_new, weights_eps, mean=0.0, std=1.0, K=K_eps)
        
        log_omega_new, x0_new, ux_new = get_log_omega(xT_new, eps_new)
        db_factor = torch.exp((log_omega_new - log_omega))  # detailed balance move factor
        
        p = torch.rand(db_factor.shape[0]).to(device)
        index_move = p < db_factor  # the index to be moved

        xT[index_move] = xT_new[index_move]
        eps[index_move] = eps_new[index_move]
        log_omega[index_move] = log_omega_new[index_move]
        x0[index_move] = x0_new[index_move]
        ux[index_move] = ux_new[index_move]
        print(i, ux.mean(), index_move.sum())
        energy.append(ux.mean().cpu())
        acceptance_numbers.append(index_move.sum().cpu())
        if  i >= 0 and i % (nmcmc//count_sava) == 0 and if_sava:
            save_mcmc_states(xT, log_omega, x0, ux, energy, acceptance_numbers, i, f'{path}-{back_coeff}-{K_x}-{K_eps}')
        if i >= nmcmc - last_steps and i % (last_steps//count_last) == 0:
            log_omega_last_steps.append(log_omega.cpu())
            x0_last_steps.append(x0.cpu())

    xT = xT.cpu()
    eps = eps.cpu()
    x0 = x0.cpu()
    ux = ux.cpu()
    state_dict = {
        'xT': xT,
        'eps': eps,
        'log_omega': log_omega,
        'x0': x0,
        'ux': ux
    }

    # Save the state dictionary to a file
    concatenated_x0_last_steps = torch.cat(x0_last_steps, dim=0)
    torch.save(state_dict, f'{path}-{back_coeff}-{K_x}-{K_eps}.pth')
    torch.save(concatenated_x0_last_steps, f'{path}-x0_last_steps-{back_coeff}-{K_x}-{K_eps}.pth')
    torch.save(energy, f'{path}-energy-{back_coeff}-{K_x}-{K_eps}.pt')
    return concatenated_x0_last_steps