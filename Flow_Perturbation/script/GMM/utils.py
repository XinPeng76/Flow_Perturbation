import torch
import numpy as np
from src.utils.common import generate_tsampling
from src.modules.EDM import cin, cnoise, cout, cskip

# 新建一个EMDSampler类，用于生成数据，输入是model和sig_data，device
class EMDSampler:
    def __init__(self, model, sig_data, device):
        self.model = model
        self.sig_data = sig_data
        self.device = device

    def sore_function(self, x, sig):
        '''
        Computes the score function for input data x at given time step t.

        The score function is a key concept in generative models, guiding the reverse process of data through a noise process.

        Parameters:
        - x (Tensor): The input data, typically noisy data at a certain time step t.
        - sig (Tensor): A scalar or a tensor with the same batch size as x, representing the time step.

        Returns:
        - Tensor: The score function value for the input data x at time step t.
        '''
        # Expand the time step t to match the batch size of x and ensure it's on the correct device
        sig = (sig * torch.ones(x.shape[0])).unsqueeze(-1).to(self.device)
        # Use the model to predict the noise for the noisy data at the given time step
        input1_F = cin(sig, self.sig_data) * x
        input2_F = cnoise(sig).squeeze(-1)
        pred_F = self.model(input1_F, input2_F)
        return cskip(sig, self.sig_data) * x + cout(sig, self.sig_data) * pred_F
    
    def heun_torch(self, tn, tn1, xtn1):
        '''
        Performs a single step of the Heun method for solving ordinary differential equations.

        Parameters:
        - tn (float): The current time step.
        - tn1 (float): The next time step.
        - xtn1 (Tensor): The data at the current time step tn.

        Returns:
        - Tensor: The estimated data at time step tn1.
        '''
        score_xtn1 = self.score_function(xtn1, tn1)
        xtn_tilde = xtn1 - (tn - tn1) * tn1 * score_xtn1
        score_xtn_tilde = self.score_function(xtn_tilde, tn)
        xtn = xtn1 - (tn - tn1) / 2 * (tn1 * score_xtn1 + tn * score_xtn_tilde)
        return xtn
    
    def exact_dynamics_heun(self, tn, tn1, xtn1):
        '''
        Performs multiple steps of the Heun method for solving ordinary differential equations.

        Parameters:
        - tn (float): The current time step.
        - tn1 (float): The next time step.
        - xtn1 (Tensor): The data at the current time step tn.

        Returns:
        - Tensor: The estimated data at time step tn1.
        '''
        ts = generate_tsampling(tn, tn1, 100, 3)
        xt = xtn1
        for i in range(len(ts) - 1, 0, -1):
            xt = self.heun_torch(ts[i - 1], ts[i], xt)
        return xt

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
    # 计算要分块的数量
    n_sample = x.shape[0]
    n_block = n_sample // block_size
    if n_sample % block_size != 0:
        n_block += 1
    # 分块计算能量
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



    
    
