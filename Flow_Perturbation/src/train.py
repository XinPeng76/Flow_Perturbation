
import torch
import numpy as np
import os
from .EDM import loss_EDM
from .DDPM import diffusion_loss_fn
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
class CustomDataset(Dataset):
    def __init__(self, features, labels1, labels2):
        self.features = features
        self.labels1 = labels1
        self.labels2 = labels2
    
    def __getitem__(self, index):
        return self.features[index], self.labels1[index], self.labels2[index]
    
    def __len__(self):
        return len(self.features)

def train_model_EDM(model, device, ndim, sig_data, dataloader, path,num_epoch=81,lr=1e-4, loss_EDM = loss_EDM,decay_steps = 20):
    if not os.path.exists(path):
        os.makedirs(path)
    seed = 1234
    print('Training model...')
    #model = MLP(ndim = ndim).to(device)  # this is F(x, t)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    for t in range(num_epoch):
        loss_list = []
        for idx,batch_x in enumerate(dataloader):
            loss = loss_EDM(model, sig_data, device, batch_x)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)	
            optimizer.step()
            loss_list.append(loss.item())
        if(t%decay_steps==0):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5			
            print(np.mean(loss_list),)
    
    torch.save(model.state_dict(), f'{path}/{ndim}-d-model.pth')
    return model

def train_model_DDPM(model, ndim, dataloader, path, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps,num_epoch=81,lr=1e-3, loss_DDPM = diffusion_loss_fn,decay_steps = 20):
    if not os.path.exists(path):
        os.makedirs(path)
    seed = 1234
    print('Training model...')

    #model = MLP(ndim = ndim).to(device)	
    #model.load_state_dict(torch.load('models/CGN/model_RotAug_LowLR.pth'))		
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    for t in range(num_epoch):
        loss_list = []

        for idx,batch_x in enumerate(dataloader):
            batch_x = batch_x
            loss = loss_DDPM(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)	
            optimizer.step()
            loss_list.append(loss.item())
        if(t%decay_steps==0):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
                param_group['lr'] = max(param_group['lr'], 1e-6)
            print(np.mean(loss_list))
            print('lr*0.5')
            torch.save(model.state_dict(), f'{path}/{ndim}-d-model_{t//100}.pth')
        
    torch.save(model.state_dict(), f'{path}/{ndim}-d-model.pth')
    
    return model

def train_model_var(model_var, dataloader, back_coeff, num_epoch=301,lr=1e-3,path='models/GMM',decay_steps = 100):
    if not os.path.exists(path):
        os.makedirs(path)
    optimizer = torch.optim.Adam(model_var.parameters(),lr=lr)
    for t in range(num_epoch):
        loss_list = []
        for idx,(x0_i, dxT_i,  eps_squarenorm_i) in enumerate(dataloader):
            forw_coeff = model_var(x0_i)
            eps_tilde = dxT_i / forw_coeff
            eps_tilde_square_norm = torch.sum(eps_tilde**2, dim=-1)

            loss = torch.mean(torch.abs(eps_tilde_square_norm - eps_squarenorm_i))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        print(t,np.mean(loss_list))
        if(t%decay_steps==0):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5			
            print(np.mean(loss_list))
    torch.save(model_var.state_dict(), f'{path}/model_var_{back_coeff}.pt')
    return model_var

def random_rotation_matrix(device):
    rotation = R.random()
    matrix = rotation.as_matrix()
    return torch.tensor(matrix, dtype=torch.float32).to(device)  # Convert to PyTorch tensor

def rotate_molecules(molecules, n_particles, n_dimensions):
    ndim = n_particles*n_dimensions
    # Ensure the input is on the correct device (CUDA) and is a float tensor
    molecules = molecules.view(-1, n_particles, n_dimensions)
    device = molecules.device
    rotation_matrix = random_rotation_matrix(device)
    rotated_molecules = rotated_molecules = torch.matmul(molecules, rotation_matrix)
    return rotated_molecules.view(-1, ndim)

def train_model_DDPM_R(model, n_particles, n_dimensions, dataloader, path, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps,num_epoch=4001, lr=1e-3, loss_DDPM = diffusion_loss_fn):
    if not os.path.exists(path):
        os.makedirs(path)
    seed = 1234
    print('Training model...')
    #model.load_state_dict(torch.load('models/CGN/model_RotAug_LowLR.pth'))		
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

    for t in range(num_epoch):
        loss_list = []

        for idx,batch_x in enumerate(dataloader):
            batch_x = rotate_molecules(batch_x, n_particles, n_dimensions)
            loss = loss_DDPM(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
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
            torch.save(model.state_dict(), f'{path}/model_RotAug_LowLR_{t//100}.pth')
        
    torch.save(model.state_dict(), f'{path}/model_RotAug_LowLR.pth')

def train_model_EDM_R(model, n_particles, n_dimensions, device, ndim, sig_data, dataloader, path,num_epoch=81,lr=1e-4, loss_EDM = loss_EDM,decay_steps = 20):
    if not os.path.exists(path):
        os.makedirs(path)
    seed = 1234
    print('Training model...')
    #model = MLP(ndim = ndim).to(device)  # this is F(x, t)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    for t in range(num_epoch):
        loss_list = []
        for idx,batch_x in enumerate(dataloader):
            batch_x = rotate_molecules(batch_x, n_particles, n_dimensions)
            loss = loss_EDM(model, sig_data, device, batch_x)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)	
            optimizer.step()
            loss_list.append(loss.item())
        if(t%decay_steps==0):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5			
            print(np.mean(loss_list),)
    
    torch.save(model.state_dict(), f'{path}/{ndim}-d-model.pth')
    return model