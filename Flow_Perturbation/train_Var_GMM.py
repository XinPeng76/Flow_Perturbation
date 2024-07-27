import torch
import numpy as np
import os
from src.common import MLP_nonorm,  MLP_var
from train_model_GMM import ndim,device,sig_data
from src.EDM import EMDSampler
from torch.utils.data import DataLoader
from src.train import train_model_var, CustomDataset
back_coeff = 0.01
tmax = 15.0
epsilon = 0.01
model = MLP_nonorm(ndim = ndim,hidden_size= 2000, hidden_layers= 10, emb_size= 80).to(device)  # this is F(x, t)
if os.path.exists(f'models/EMD/{ndim}-d-model.pth'):			
    model.load_state_dict(torch.load(f'models/GMM/{ndim}-d-model.pth'))
else:
    OSError('No model found, please train the model first!')
model.eval()
for param in model.parameters():
    param.requires_grad = False
Sampler = EMDSampler(model, sig_data, device)

exact_dynamics_heun = Sampler.exact_dynamics_heun
if __name__ == '__main__':
    if not os.path.exists('models/GMM'):
        os.makedirs('models/GMM')
    sampN = 10000
    lambdaT = tmax**2

    xT = np.sqrt(lambdaT) * torch.randn(sampN, ndim).to(device)
    eps = torch.randn_like(xT)
    x0 = exact_dynamics_heun(epsilon, tmax, xT) + back_coeff * eps # backward dynamics
    #print(get_energy_device(x0,Nwellinfo))

    dxT = xT - exact_dynamics_heun(tmax, epsilon, x0) # this is the error in xT that needs to be matched
    eps_squarenorm = torch.sum(eps**2, dim=-1)

    batch_size = 128
    dataset = CustomDataset(x0, dxT, eps_squarenorm)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True, drop_last=True)

    # Create the model
    model_var = MLP_var(ndim=ndim).to(device)
    model_var = train_model_var(model_var, dataloader, back_coeff, num_epoch=301,lr=1e-3,path='models/GMM',decay_steps = 100)
