import torch
import numpy as np
import os
from src.common import MLP_nonorm,  MLP_var
from train_model_GMM import ndim,device,sig_data
from utils import EMDSampler
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
    #print(dxT)
    from torch.utils.data import Dataset, DataLoader
    class CustomDataset(Dataset):
        def __init__(self, features, labels1, labels2):
            self.features = features
            self.labels1 = labels1
            self.labels2 = labels2
        
        def __getitem__(self, index):
            return self.features[index], self.labels1[index], self.labels2[index]
        
        def __len__(self):
            return len(self.features)

    batch_size = 128
    dataset = CustomDataset(x0, dxT, eps_squarenorm)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True, drop_last=True)

    # Create the model
    model_var = MLP_var(ndim=ndim).to(device)
    num_epoch = 301
    optimizer = torch.optim.Adam(model_var.parameters(),lr=1e-3)

    for t in range(num_epoch):
        loss_list = []
        for idx,(x0_i, dxT_i,  eps_squarenorm_i) in enumerate(dataloader):
            #print(x0_i.dtype)
            forw_coeff = model_var(x0_i)
            #print(forw_coeff.shape)
            eps_tilde = dxT_i / forw_coeff
            eps_tilde_square_norm = torch.sum(eps_tilde**2, dim=-1)

            loss = torch.mean(torch.abs(eps_tilde_square_norm - eps_squarenorm_i))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            #print(log_omega)
        print(t,np.mean(loss_list))
        
        
        if(t%100==0):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5			
            print(np.mean(loss_list))

    torch.save(model_var.state_dict(), 'models/GMM/model_var_{}.pt'.format(back_coeff))