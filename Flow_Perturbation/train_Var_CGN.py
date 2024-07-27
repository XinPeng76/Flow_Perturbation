import torch
import numpy as np
import os
from src.common import MLP,  MLP_var
from src.utils import remove_mean, generate_tsampling
from train_model_CGN import alphas_prod,n_dimensions,n_particles,ndim,device,num_steps
from src.DDPM import interpolate_parameters,DDPMSamplerCoM
from torch.utils.data import DataLoader
from src.train import train_model_var, CustomDataset

back_coeff = 0.001
model = MLP(ndim=ndim).to(device)
if os.path.exists('models/CGN/model_RotAug_LowLR.pth'):			
    model.load_state_dict(torch.load('models/CGN/model_RotAug_LowLR.pth'))
else:
    OSError('No model found, please train the model first!')

model.eval()
for param in model.parameters():
    param.requires_grad = False
st, sigma_t, st_derivative, sigma_t_derivative = interpolate_parameters(num_steps, alphas_prod)

Sampler = DDPMSamplerCoM(model, st, st_derivative, sigma_t_derivative, n_particles, n_dimensions, device)

ddpm_ode_heun = Sampler.exact_dynamics_heun

time_forward = generate_tsampling(1, num_steps-1, 100, 2.0)
time_backward = time_forward[::-1]

if __name__ == '__main__':
    if not os.path.exists('models/CGN'):
        os.makedirs('models/CGN')
    x0_list = []
    dxT_list = []
    eps_squarenorm_list = []

    sampN = 2000
    for i in range(50):
        xT =  torch.randn(sampN, ndim).to(device)
        xT = remove_mean(xT, n_particles, n_dimensions)
        eps = torch.randn_like(xT)
        eps = remove_mean(eps, n_particles, n_dimensions)

        x0 = ddpm_ode_heun(xT, time_backward) + back_coeff * eps # backward dynamics

        dxT = xT -  ddpm_ode_heun(x0, time_forward)# this is the error in xT that needs to be matched
        eps_squarenorm = torch.sum(eps**2, dim=-1)
        
        x0_list.append(x0.cpu())
        dxT_list.append(dxT.cpu())
        eps_squarenorm_list.append(eps_squarenorm.cpu())

    x0 = torch.cat(x0_list).to(device)
    dxT = torch.cat(dxT_list).to(device)
    eps_squarenorm = torch.cat(eps_squarenorm_list).to(device)

    batch_size = 128
    dataset = CustomDataset(x0, dxT, eps_squarenorm)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True, drop_last=True)

    # Create the model
    model_var = MLP_var(ndim=ndim).to(device)
    model_var = train_model_var(model_var, dataloader, back_coeff, num_epoch=301,lr=1e-3,path='models/CGN',decay_steps = 100)