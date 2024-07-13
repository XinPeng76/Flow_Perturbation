import torch
import os
from bgmol.datasets import ChignolinOBC2PT
import numpy as np
import torch.nn as nn
from scipy.spatial.transform import Rotation as R
from src.common import MLP
from src.utils import remove_mean
from src.DDPM import calc_alphas_betas, diffusion_loss_fn


# Constants
n_dimensions = 3 
n_particles = 175
ndim = n_particles * n_dimensions

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float32
is_data_here = os.path.isdir("ChignolinOBC2PT")
dataset = ChignolinOBC2PT(download=not is_data_here, read=True)

system = dataset.system
coordinates = dataset.coordinates
temperature = dataset.temperature

all_data = coordinates.reshape(-1, dataset.dim)

dataset1 = torch.tensor(all_data).float()
dataset1 = remove_mean(dataset1,n_particles, n_dimensions)

num_steps = 1000
alphas, betas, alphas_prod, alphas_bar_sqrt, one_minus_alphas_bar_sqrt = calc_alphas_betas(num_steps=num_steps, scaling=10, beta_min=1e-5, beta_max=1e-2)

if __name__ == '__main__':
    if not os.path.exists('models/CGN'):
        os.makedirs('models/CGN')
    seed = 1234
    print('Training model...')
    batch_size = 256
    dataset1 = dataset1.to(device)
    print(dataset1.shape)
    dataloader = torch.utils.data.DataLoader(dataset1,batch_size=batch_size,shuffle=True)

    model = MLP(ndim = ndim).to(device)	
    #model.load_state_dict(torch.load('models/CGN/model_RotAug_LowLR.pth'))		
    num_epoch = 4001
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    def random_rotation_matrix():
        rotation = R.random()
        matrix = rotation.as_matrix()
        return torch.tensor(matrix, dtype=torch.float32).to(device)  # Convert to PyTorch tensor

    def rotate_molecules(molecules):
        # Ensure the input is on the correct device (CUDA) and is a float tensor
        molecules = molecules.view(-1, n_particles, n_dimensions)
        
        rotation_matrix = random_rotation_matrix()
        rotated_molecules = rotated_molecules = torch.matmul(molecules, rotation_matrix)
        return rotated_molecules.view(-1, ndim)

    for t in range(num_epoch):
        loss_list = []

        for idx,batch_x in enumerate(dataloader):
            batch_x = rotate_molecules(batch_x)
            loss = diffusion_loss_fn(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)	
            optimizer.step()
            loss_list.append(loss.item())
        if(t%50==0):	
            print(np.mean(loss_list))
        if(t%500==0):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
                param_group['lr'] = max(param_group['lr'], 1e-6)
            print('lr*0.5')
            torch.save(model.state_dict(), 'models/CGN/model_RotAug_LowLR_{}.pth'.format(t//100))
        
    torch.save(model.state_dict(), 'models/CGN/model_RotAug_LowLR.pth')

