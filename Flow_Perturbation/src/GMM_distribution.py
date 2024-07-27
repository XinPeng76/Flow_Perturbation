import torch
import numpy as np

def sample_NWell(nsamples, ndim, nwell):
    # this produce equal amount of samples (nsamples) in each well
    # there are total of nwell wells. the dimension of each well is ndim

    # Create a local random number generator with a specific seed
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(43)
    rng3 = np.random.default_rng(45)
    # Generate location of wells using rng1
    mus = rng1.normal(size=(nwell, ndim))
    # Generate width of wells using rng2
    sigmas = 0.4+np.abs(rng2.normal(loc = 0.1, scale=0.5, size=(nwell, ndim)))
    # Generate the coefficients for each well using rng3
    coeffs = np.abs(rng3.normal(loc=0.1, scale=2, size=(nwell)))

    samples = np.zeros((nwell, nsamples, ndim))
    for i in range(nwell):
        samples[i] = np.random.multivariate_normal(mus[i], np.diag(sigmas[i]), nsamples)

    return samples, [mus, sigmas, coeffs]

def get_energy_device_0(x, Nwell_info):
    ndim = x.shape[-1]
    mus, sigmas, coeffs = Nwell_info
    mus = mus.to(x.device)
    sigmas = sigmas.to(x.device)
    coeffs = coeffs.to(x.device)
    exponent_sample_to_well = torch.sum((x.unsqueeze(-2) - mus)**2 / (2*sigmas), dim=-1) - torch.log(coeffs) + 0.5*torch.sum(torch.log(sigmas), dim=-1) # shape (nsample, ndim)
    exponent_minvalue, exponent_minindex = torch.min(exponent_sample_to_well, dim=-1) # shape (nsample)
    
    ux = exponent_minvalue
    
    exponent_sample_to_well_relative = exponent_sample_to_well - exponent_minvalue.unsqueeze(-1) # shape (nsample, ndim)
    
    ux_correction = torch.log(torch.sum(torch.exp(-exponent_sample_to_well_relative), dim=-1))
    
    ux = ux + ux_correction + torch.log(torch.sum(coeffs)) + ndim/2*np.log(2*np.pi)
    
    return ux

def get_energy_device(x, Nwell_info, block_size=10000):
    ndim = x.shape[-1]
    mus, sigmas, coeffs = Nwell_info
    mus = mus.to(x.device)
    sigmas = sigmas.to(x.device)
    coeffs = coeffs.to(x.device)
    n_sample = x.shape[0]
    n_block = n_sample // block_size
    if n_sample % block_size != 0:
        n_block += 1
    ux = torch.zeros(n_sample).to(x.device)
    for i in range(n_block):
        start = i * block_size
        end = min((i+1) * block_size, n_sample)
        x_block = x[start:end]
        mus_block = mus.unsqueeze(0).expand(end-start, -1, -1)
        sigmas_block = sigmas.unsqueeze(0).expand(end-start, -1, -1)
        coeffs_block = coeffs.unsqueeze(0).expand(end-start, -1)
        exponent_sample_to_well = torch.sum((x_block.unsqueeze(-2) - mus_block)**2 / (2*sigmas_block), dim=-1) - torch.log(coeffs_block) + 0.5*torch.sum(torch.log(sigmas_block), dim=-1) # shape (nsample, ndim)
        exponent_minvalue, exponent_minindex = torch.min(exponent_sample_to_well, dim=-1) # shape (nsample)

        ux[start:end] = exponent_minvalue

        exponent_sample_to_well_relative = exponent_sample_to_well - exponent_minvalue.unsqueeze(-1) # shape (nsample, ndim)

        ux_correction = torch.log(torch.sum(torch.exp(-exponent_sample_to_well_relative), dim=-1))

        ux[start:end] = ux[start:end] + ux_correction + torch.log(torch.sum(coeffs_block, dim=-1)) + ndim/2*np.log(2*np.pi)

    return ux