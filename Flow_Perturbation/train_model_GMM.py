import torch
import os
import numpy as np
from utils import sample_NWell
from src.common import MLP_nonorm
from src.EDM import loss_EDM

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
    if not os.path.exists('models/GMM'):
        os.makedirs('models/GMM')
    seed = 1234
    print('Training model...')
    batch_size = 128
    dataset = dataset.to(device)
    print(dataset.shape)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)
    num_epoch = 81

    model = MLP_nonorm(ndim = 1000, hidden_size= 2000, hidden_layers= 10, emb_size= 80).to(device)  # this is F(x, t)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    for t in range(num_epoch):
        loss_list = []
        for idx,batch_x in enumerate(dataloader):
            loss = loss_EDM(model, sig_data, device, batch_x)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)	
            optimizer.step()
            loss_list.append(loss.item())
        if(t%20==0):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5			
            print(np.mean(loss_list))
    
    torch.save(model.state_dict(), f'models/GMM/{ndim}-d-model.pth')