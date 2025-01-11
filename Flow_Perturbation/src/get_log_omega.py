import torch
import numpy as np

def get_log_omega_FP(xT,eps,exact_dynamics,time_backward,get_energy,tmax=1.0):
    sampN,ndim = xT.shape
    time_forward = time_backward[::-1]
    back_coeff = 0.001
    dx0 = back_coeff * eps
    x0 = exact_dynamics(xT, time_backward)  # backward dynamics
    x0p = x0 + dx0
    x0n = x0 - dx0

    x0_all = torch.cat([x0p, x0n], dim=0)
    xT_all = torch.cat([xT, xT], dim=0)
    dxT_all = xT_all - exact_dynamics(x0_all, time_forward)
    dxT = (dxT_all[:sampN] - dxT_all[sampN:])/2

    dx0_normsquare = torch.sum(dx0**2, dim=-1)
    dxT_normsquare = torch.sum(dxT**2, dim=-1)

    alpha_epsz_square = dxT_normsquare / dx0_normsquare # the streching factor
    #deltaSt =  - ndim/2 * torch.log(alpha_epsz_square) + np.log(gamma(ndim/2) / (np.pi**(ndim/2)*2))
    deltaSt =  - ndim/2 * torch.log(alpha_epsz_square)
    uz = torch.sum(xT**2/tmax**2, dim=-1)/2.0  + 0.5*ndim*np.log(2*np.pi) + ndim*np.log(tmax)
    ux = get_energy(x0).reshape(-1)
    log_omega = -ux + deltaSt + uz

    return log_omega, x0, ux

def get_log_omega_J(xT, eps, exact_dynamics_dSt, time_backward, get_energy, tmax=1.0, nnoise=1,method = 'RK4',eps_type='Rademacher'):
    ndim = xT.shape[-1]
    x0, deltaSt = exact_dynamics_dSt(xT, time_backward, method = method, nnoise = nnoise, eps_type = eps_type)  
    uz = torch.sum(xT**2/tmax**2, dim=-1)/2.0  + 0.5*ndim*np.log(2*np.pi) + ndim*np.log(tmax)
    ux = get_energy(x0).reshape(-1)
    log_omega = -ux + deltaSt + uz
    
    return log_omega, x0, ux

def get_log_omega_SNF(xT, eps, exact_dynamics_dSt, time_backward, get_energy, langevin_layer, tmax=1.0, method = 'RK4'):
    ndim = xT.shape[-1]
    x0, deltaSt = exact_dynamics_dSt(xT, time_backward, method = method,nnoise = -1)
    x0_next, log_prob_ratio = langevin_layer(x0)
    uz = torch.sum(xT**2/tmax**2, dim=-1)/2.0  + 0.5*ndim*np.log(2*np.pi) + ndim*np.log(tmax)
    ux = get_energy(x0_next).reshape(-1)
    log_omega = -ux + deltaSt + uz + log_prob_ratio
    
    return log_omega, x0, ux