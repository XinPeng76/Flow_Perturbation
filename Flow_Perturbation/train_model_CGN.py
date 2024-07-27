import torch
import os
from bgmol.datasets import ChignolinOBC2PT
import numpy as np
import torch.nn as nn
from scipy.spatial.transform import Rotation as R
from src.common import MLP
from src.utils import remove_mean
from src.DDPM import calc_alphas_betas, diffusion_loss_fn
from src.train import train_model_DDPM_R
def random_rotation_matrix():
    rotation = R.random()
    matrix = rotation.as_matrix()
    return torch.tensor(matrix, dtype=torch.float32).to(device)  # Convert to PyTorch tensor

def rotate_molecules(molecules, n_particles, n_dimensions):
    # Ensure the input is on the correct device (CUDA) and is a float tensor
    molecules = molecules.view(-1, n_particles, n_dimensions)
    
    rotation_matrix = random_rotation_matrix()
    rotated_molecules = rotated_molecules = torch.matmul(molecules, rotation_matrix)
    return rotated_molecules.view(-1, ndim)

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
target_energy = dataset.get_energy_model(n_simulation_steps=0)

all_data = coordinates.reshape(-1, dataset.dim)

dataset1 = torch.tensor(all_data).float()
dataset1 = remove_mean(dataset1,n_particles, n_dimensions)

num_steps = 1000
alphas, betas, alphas_prod, alphas_bar_sqrt, one_minus_alphas_bar_sqrt = calc_alphas_betas(num_steps=num_steps, scaling=10, beta_min=1e-5, beta_max=1e-2)

if __name__ == '__main__':
    batch_size = 256
    dataset1 = dataset1.to(device)
    print(dataset1.shape)
    dataloader = torch.utils.data.DataLoader(dataset1,batch_size=batch_size,shuffle=True)
    model = MLP(ndim = ndim).to(device)
    train_model_DDPM_R(model, n_particles, n_dimensions, dataloader, 'models/CGN', alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)