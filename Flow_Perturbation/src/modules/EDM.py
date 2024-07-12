import torch
import torch.nn as nn

def generate_sig(mean, std_dev, batchsize):
    normal_dist = torch.normal(mean, std_dev, size=(batchsize,))
    log_normal_dist = torch.exp(normal_dist)
    return log_normal_dist

def cskip(sig, sig_data): # the coeff of x in the D(x, t) expression
    return sig_data**2/(sig**2 + sig_data**2)

def cout(sig, sig_data): # the coeff that scales F(x, t)
    return sig * sig_data / torch.sqrt(sig**2 + sig_data**2)

def cin(sig, sig_data): # the term that scales x inside F(x, t)
    return 1/torch.sqrt(sig**2 + sig_data**2)

def cnoise(sig): # the terms that scales the t term in F(x,t)
    return 1/4*torch.log(sig)

def loss_EDM(model, sig_data,device, batch_x):
    batch_size = batch_x.shape[0]
    y = batch_x
    sig = generate_sig(-1.2, 1.2, batch_size).unsqueeze(-1).to(device) # shape is (batch_size, 1)
    e = torch.randn_like(y).to(device) # shape is (batch_size, ndim)
    n = sig * e # scale the variance
    input1_F = cin(sig, sig_data) * (y + n)
    input2_F = cnoise(sig).squeeze(-1) # shape is (batch_size,)

    pred_F = model(input1_F, input2_F)

    effective_target = 1.0/cout(sig, sig_data)*(y - cskip(sig, sig_data)*(y + n))

    loss = (pred_F - effective_target).square().mean()
    return loss


