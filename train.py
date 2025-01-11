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
from Flow_Perturbation.src.utils import  get_new_log_dir,get_logger,remove_mean, clean_up
from Flow_Perturbation.src.SMC import generate_doubling_intervals_exclude_start,systematic_resampling,find_closest_larger_element_desc,mc_step,resample_if_needed
from Flow_Perturbation.src.get_log_omega import get_log_omega_FP,get_log_omega_J,get_log_omega_SNF
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('path', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume_iter', type=int, default=None)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--nsamples', type=int, default=200000)
    parser.add_argument('--if_v', type=bool, default=True)
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
    
    if config.dataset.name == 'GMM':
        samples, Nwellinfo, mvn_list = sample_NWell(args.nsamples, config.model.ndim, config.model.nwell)
        dataset = samples.reshape(-1, config.model.ndim).to(args.device)
    elif config.dataset.name == 'CGN':
        from bgmol.datasets import ChignolinOBC2PT
        is_data_here = os.path.isdir("ChignolinOBC2PT")
        CGN = ChignolinOBC2PT(download=not is_data_here, read=True)
        dataset = CGN.coordinates.reshape(-1, CGN.dim)
        n_dimensions = 3 
        n_particles = 175
        dataset = remove_mean(torch.tensor(dataset).float(),n_particles, n_dimensions).to(args.device)
    else:
        raise ValueError('Dataset not implemented')
    
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=config.train.batch_size,shuffle=True)
    logger = get_logger(config.dataset.name, log_dir, 'train.log')
    logger.info(args)
    logger.info(config)
    alphas, betas, alphas_prod, alphas_bar_sqrt, one_minus_alphas_bar_sqrt =\
          calc_alphas_betas(num_steps=config.model.num_steps, scaling=config.model.scaling, beta_min=config.model.beta_min, beta_max=config.model.beta_max)
    if config.train.if_norm:
        model = MLP(ndim=config.model.ndim,hidden_size=config.model.hidden_size,hidden_layers=config.model.hidden_layers,emb_size=config.model.emb_size).to(args.device)
    else:
        model = MLP_nonorm(ndim=config.model.ndim,hidden_size=config.model.hidden_size,hidden_layers=config.model.hidden_layers,emb_size=config.model.emb_size).to(args.device)
    model = train_model_DDPM(model, config.model.ndim, dataloader, args.path, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, \
                             config.model.num_steps,num_epoch=config.train.num_epoch,lr=config.train.lr, loss_DDPM = diffusion_loss_fn_v_prediction,decay_steps = config.train.decay_steps)
    clean_up()