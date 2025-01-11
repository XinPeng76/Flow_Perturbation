import numpy as np
import os
import shutil
import argparse
import yaml
from easydict import EasyDict
from glob import glob
import torch
from Flow_Perturbation.src.GMM_distribution import get_energy_device,sample_NWell,redraw_samples,get_energy_gradient_device
from Flow_Perturbation.src.DDPM import calc_alphas_betas, diffusion_loss_fn, diffusion_loss_fn_v_prediction
from Flow_Perturbation.src.common import MLP_nonorm,LangevinDynamicsWithLogP,MLP
from Flow_Perturbation.src.train import train_model_DDPM
from Flow_Perturbation.src.DDPM import interpolate_parameters,DDPMSamplerCoM, DDPMSampler
from Flow_Perturbation.src.utils import  generate_tsampling,get_new_log_dir,get_beta_schedule,generate_K_values,get_logger, clean_up
from Flow_Perturbation.src.SMC import mc_step,resample_if_needed,dists5_ratio,x0_ratio
from Flow_Perturbation.src.get_log_omega import get_log_omega_FP,get_log_omega_J,get_log_omega_SNF
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('path', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume_iter', type=int, default=None)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--nsamples', type=int, default=10)
    parser.add_argument('--if_v', type=bool, default=True)
    parser.add_argument('--tmax', type=float, default=1.0)
    parser.add_argument('--sampN', type=int, default=1000)
    parser.add_argument('--eps_type', type=str, default='Rademacher', help='Rademacher or Gaussian')
    parser.add_argument('--method', type=int, default=0, help='0: FP, -1: Jocobian, -2: SNF , 1-n: Hutchinson')
    args = parser.parse_args()
    resume = os.path.isdir(args.config)
    if resume:
        config_path = glob(os.path.join(args.config, '*.yml'))[0]
        resume_from = args.config
    else:
        config_path = args.config

    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]

    # Logging
    if resume:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag='resume')
        os.symlink(os.path.realpath(resume_from), os.path.join(log_dir, os.path.basename(resume_from.rstrip("/"))))
    else:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name)
        #shutil.copytree('./models', os.path.join(log_dir, 'models'))

    logger = get_logger(config.dataset.name, log_dir, str(args.method)+'.log')
    logger.info(args)
    logger.info(config)

    if config.dataset.name == 'GMM':
        _, Nwellinfo, mvn_list = sample_NWell(args.nsamples, config.model.ndim, config.model.nwell)
        Nwellinfo_device = []
        for i in range(len(Nwellinfo)):
            Nwellinfo_device.append(Nwellinfo[i].to(args.device))
        mvn_list_device = []
        for mvn in mvn_list:
            mvn_device = torch.distributions.MultivariateNormal(
                mvn.loc.to(args.device), mvn.covariance_matrix.to(args.device)
            )
            mvn_list_device.append(mvn_device)
        n_dimensions = 1 
        n_particles = config.model.ndim
        get_energy = lambda x: get_energy_device(x, mvn_list_device, Nwellinfo_device)
        potential_fn = lambda x: get_energy_gradient_device(x, mvn_list_device, Nwellinfo_device)
        if_com = False
        RC_ratio = lambda x: x0_ratio(x, threshold=0.0)
        time_forward = generate_tsampling(0, config.model.num_steps-1, config.SMC.ode_steps, 2.0)
    elif config.dataset.name == 'CGN':
        from bgmol.datasets import ChignolinOBC2PT
        is_data_here = os.path.isdir("ChignolinOBC2PT")
        CGN = ChignolinOBC2PT(download=not is_data_here, read=True)
        n_dimensions = 3 
        n_particles = config.model.ndim//3
        get_energy = lambda x: CGN.get_energy_model(n_simulation_steps=0).energy(x)
        potential_fn = lambda x: CGN.get_energy_model(n_simulation_steps=0).force(x)
        if_com = True
        RC_ratio = lambda x: dists5_ratio(x, n_particles=n_particles, n_dimensions=n_dimensions)
        time_forward = generate_tsampling(1, config.model.num_steps-1, config.SMC.ode_steps, 2.0)
    else:
        raise ValueError('Dataset not implemented')

    time_backward = time_forward[::-1]
    alphas, betas, alphas_prod, alphas_bar_sqrt, one_minus_alphas_bar_sqrt =\
          calc_alphas_betas(num_steps=config.model.num_steps, scaling=config.model.scaling, beta_min=config.model.beta_min, beta_max=config.model.beta_max)
    st, sigma_t, st_derivative, sigma_t_derivative,at,oat = interpolate_parameters(config.model.num_steps, alphas_prod,alphas_bar_sqrt, one_minus_alphas_bar_sqrt)
    if config.train.if_norm:
        model = MLP(ndim=config.model.ndim,hidden_size=config.model.hidden_size,hidden_layers=config.model.hidden_layers,emb_size=config.model.emb_size).to(args.device)
    else:
        model = MLP_nonorm(ndim=config.model.ndim,hidden_size=config.model.hidden_size,hidden_layers=config.model.hidden_layers,emb_size=config.model.emb_size).to(args.device)
    model.load_state_dict(torch.load(f'{args.path}/model.pth', map_location=args.device,weights_only=False))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    if if_com :
        Sampler = DDPMSamplerCoM(model, st, st_derivative, sigma_t_derivative,at,oat, n_particles, n_dimensions,args.if_v,args.device)
    else:
        Sampler = DDPMSampler(model, st, st_derivative, sigma_t_derivative,at,oat,args.if_v,args.device)
    exact_dynamics = Sampler.exact_dynamics
    exact_dynamics_dSt = Sampler.exact_dynamics_dSt
    langevin_layer = LangevinDynamicsWithLogP( 0.01, 1.0, potential_fn)
    if args.method == 0:
        get_log_omega = lambda xT, eps: get_log_omega_FP(xT, eps, exact_dynamics, time_backward, get_energy, tmax=args.tmax)
        if_K_eps = True
        logger.info(f'method==,{args.method}, using method FP')
        # log_dir加上FP
    elif args.method == -1:
        get_log_omega = lambda xT, eps: get_log_omega_J(xT, eps, exact_dynamics_dSt, time_backward, get_energy, tmax=args.tmax, nnoise=args.method)
        if_K_eps = False
        logger.info(f'method==, {args.method}, using method J')
    elif args.method == -2:
        get_log_omega = lambda xT, eps: get_log_omega_SNF(xT, eps, exact_dynamics_dSt, time_backward, get_energy, langevin_layer, tmax=args.tmax)
        if_K_eps = False
        logger.info(f'method==, {args.method}, using method SNF')
    elif args.method > 0:
        get_log_omega = lambda xT, eps: get_log_omega_J(xT, eps, exact_dynamics_dSt, time_backward, get_energy, tmax=args.tmax, nnoise=args.method, eps_type=args.eps_type)
        if_K_eps = False
        logger.info(f'method==, {args.method}, using method Hutchinson')
    else:
        raise ValueError('method should be 0, -1, -2 or >0, 0: FP, -1: Jocobian, -2: SNF , 1-n: Hutchinson')


    xT_init = args.tmax * torch.randn(config.SMC.n_replicas, config.model.ndim).to(args.device)
    eps_init = torch.randn_like(xT_init)

    log_omega_init, x0_init,ux_init = get_log_omega(xT_init, eps_init)

    beta = get_beta_schedule(config.SMC.n_steps, beta_cut = config.SMC.beta_cut)
    K_x, K_eps = generate_K_values(config.SMC.n_steps, config.SMC.Kmax_x, config.SMC.Kmax_eps, \
                                   config.SMC.Kmin_x, config.SMC.Kmin_eps, config.SMC.compress_exp_x, config.SMC.compress_exp_eps, args.device)
    if config.SMC.beta_cut > 0:
        print('load MC')
        xT_init = torch.load(f'{config.dataset.MC}/results.pth',weights_only=False)['xT']
        xT_init = torch.stack(xT_init, dim=0).reshape(-1,config.model.ndim)[-config.SMC.n_replicas:].to(args.device)
    log_omega_init, x0_init,ux_init = get_log_omega(xT_init, eps_init)
    xT = xT_init.clone()
    eps = eps_init.clone()
    x0 = x0_init.clone()
    log_omega = log_omega_init.clone()
    ux = ux_init.clone()
    #print(ux.mean())

    total_logweight = torch.zeros_like(log_omega)
    total_logweight += (beta[1] - beta[0]) * log_omega

    accept_rate = torch.zeros_like(beta)

    ansestors = torch.arange(config.SMC.n_replicas).to(args.device)
    results = {'ux_mean': [ux.clone().mean().cpu().item()], 'x0': [x0.clone().cpu().numpy()], 'ansestors': [ansestors.clone().cpu().numpy()]}
    for i in range(1, config.SMC.n_steps-1):
        betai = beta[i]
        betaip1 = beta[i+1]
        Ki_x = int(K_x[i].item())
        Ki_eps = int(K_eps[i].item())
        # mc_step
        xT, eps, log_omega, x0, ux, accept_rate[i] = mc_step(xT, eps, log_omega, x0, ux, Ki_x, Ki_eps, get_log_omega, betai, nmc=config.SMC.nmc, if_K_eps=if_K_eps, \
                                                              if_com = if_com, n_particles = n_particles, n_dimensions = n_dimensions)
        # resample
        total_logweight += (betaip1 - betai) * log_omega
        # this part do the resampling
        weights = torch.softmax(total_logweight, dim=0)  # get the normalized weights
        ess = 1 / torch.sum(weights**2) # get the effective sample size
        xT, eps, log_omega, x0, ux, ansestors, total_logweight = resample_if_needed(ess, config.SMC.n_replicas, i, config.SMC.n_steps, xT, eps, log_omega, x0, ux, \
                                                                                     ansestors, total_logweight, weights,ess_threshold=0.95)
        # compute the mean of the potential energy
        unique_elements, counts = torch.unique(ansestors, return_counts=True)
        results['ux_mean'].append(ux.clone().mean().cpu().item())
        #results['x0'].append(x0.clone().cpu().numpy())
        results['ansestors'].append(ansestors.clone().cpu().numpy())
        if i %1 == 0:
            logger.info(f'Step {i}/{config.SMC.n_steps}: Ki_x={Ki_x}, Ki_eps={Ki_eps}')
            logger.info(f'ux_mean: {ux.mean().item()}, mc_acc_rate: {accept_rate[i].item()}, RC_ratio: {RC_ratio(x0)}, unique_elements: {unique_elements.shape[0]}')
    
    state_dict = {
    'xT': xT,
    'eps': eps,
    'log_omega': log_omega,
    'x0': x0,
    'ux': ux
    }
    # save the results
    torch.save(results, os.path.join(log_dir, 'results.pth'))
    torch.save(state_dict, os.path.join(log_dir, 'state_dict.pth'))
    torch.save(accept_rate.cpu(), os.path.join(log_dir, 'accept_rate.pth'))
    logger.info(f'x0 rate:, {(x0[:, 0] < 0.0).sum().item()/config.SMC.n_replicas}')
    logger.info(f'unique elements:, {unique_elements.shape[0]}')
    logger.info(f'ux mean:, {ux.mean().item()}')
    logger.info('done')
    clean_up()