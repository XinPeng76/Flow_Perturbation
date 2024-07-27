import torch
import os
import numpy as np
from src.GMM_distribution import sample_NWell
from src.common import MLP_nonorm
from src.EDM import loss_EDM
from src.train import train_model_EDM

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

nsamples = 20000
ndim = 1000
nwell = 10
samples, Nwellinfo = sample_NWell(nsamples, ndim, nwell)
Nwellinfo[-1] = np.ones_like(Nwellinfo[-1])

dataset = torch.Tensor(samples).float().reshape((-1,ndim))
dataset = dataset.to(device)
sig_data = 1.4
print('sig_data:',sig_data)

Nwellinfo = [torch.tensor(array, dtype=torch.float).to(device) for array in Nwellinfo]

if __name__ == '__main__':
    model = MLP_nonorm(ndim = ndim, hidden_size=2000, hidden_layers=10, emb_size=80).to(device)  # this is F(x, t)
    path = 'models/GMM'
    batch_size = 128
    dataset = dataset.to(device)
    print(dataset.shape)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)
    train_model_EDM(model, device, ndim, sig_data, dataloader, path, loss_EDM = loss_EDM)
    print('Training done!')