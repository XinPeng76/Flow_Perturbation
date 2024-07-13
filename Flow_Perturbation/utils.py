import torch
import numpy as np
from src.utils import generate_tsampling
from src.EDM import cin, cnoise, cout, cskip

class EMDSampler:
    def __init__(self, model, sig_data, device):
        self.model = model
        self.sig_data = sig_data
        self.device = device

    def ideal_denoiser(self, x, sig_data, sig):
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
        input1_F = cin(sig, sig_data) * x
        input2_F = cnoise(sig).squeeze(-1)
        pred_F = self.model(input1_F, input2_F)
        return cskip(sig, sig_data) * x + cout(sig, sig_data) * pred_F
    def score_function(self, x, sig):
        return (self.ideal_denoiser(x, self.sig_data, sig) - x)/sig**2

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
            #print(ts[i-1], ts[i])
            xt = self.heun_torch(ts[i-1], ts[i], xt)
            #print(xt)
        return xt
from src.utils import remove_mean

class DDPMSamplerCOM:
    def __init__(self, model, st, st_derivative, sigma_t_derivative, n_particles, n_dimensions , device = 'cuda:0'):
        self.model = model
        self.st = st
        self.st_derivative = st_derivative
        self.sigma_t_derivative = sigma_t_derivative
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.device = device

    def score_function(self, x, t):
        '''
        Computes the score function for input data x at given time step t.

        The score function is a key concept in generative models, guiding the reverse process of data through a noise process.

        Parameters:
        - x (Tensor): The input data, typically noisy data at a certain time step t.
        - t (Tensor): A scalar or a tensor with the same batch size as x, representing the time step.

        Returns:
        - Tensor: The score function value for the input data x at time step t.
        '''
        # Expand the time step t to match the batch size of x and ensure it's on the correct device
        t_repeat = (t * torch.ones(x.shape[0])).to(self.device)
        # Use the model to predict the noise for the noisy data at the given time step
        pred_noise = self.model(x, t_repeat)
        # Remove the mean of the predicted noise
        pred_noise = remove_mean(pred_noise, self.n_particles, self.n_dimensions)
        # Calculate the value of the score function
        return self.st_derivative(t) / self.st(t) * x + self.st(t) * self.sigma_t_derivative(t) * pred_noise

    def heun_torch(self, xt, t, t_next):
        '''
        Integrates the data at a given time step using the Heun method to estimate the data at the next time step.

        The Heun method, an improved version of the Euler method, increases estimation accuracy by averaging the slopes at the current and next time steps.

        Parameters:
        - xt (Tensor): The data at the current time step t.
        - t (float): The current time step.
        - t_next (float): The next time step.

        Returns:
        - Tensor: The estimated data at time step t_next.
        '''
        # Calculate the slope at the current time step t
        dx = self.score_function(xt, t) * (t_next - t)
        # Estimate the data at the next time step using the current slope
        xt_tilde = xt + dx

        # Calculate the slope at the estimated next time step t_next
        dx_tilde = self.score_function(xt_tilde, t_next) * (t_next - t)

        # Update the data using the average of the current and estimated next slopes
        return xt + (dx + dx_tilde) / 2

    def exact_dynamics_heun(self, xT, timesteps): # do a lot of Heun steps between tn and tn1 to get the accurate xtn
        xt = remove_mean(xT, self.n_particles, self.n_dimensions)
        for i in range(len(timesteps)-1):
            t = timesteps[i]
            tnext = timesteps[i+1]
            xt = self.heun_torch(xt, t, tnext)
            xt = remove_mean(xt, self.n_particles, self.n_dimensions)
        return xt
    
class DDPMSampler:
    def __init__(self, model, st, st_derivative, sigma_t_derivative, device = 'cuda:0'):
        self.model = model
        self.st = st
        self.st_derivative = st_derivative
        self.sigma_t_derivative = sigma_t_derivative
        self.device = device

    def score_function(self, x, t):
        '''
        Computes the score function for input data x at given time step t.

        The score function is a key concept in generative models, guiding the reverse process of data through a noise process.

        Parameters:
        - x (Tensor): The input data, typically noisy data at a certain time step t.
        - t (Tensor): A scalar or a tensor with the same batch size as x, representing the time step.

        Returns:
        - Tensor: The score function value for the input data x at time step t.
        '''
        # Expand the time step t to match the batch size of x and ensure it's on the correct device
        t_repeat = (t * torch.ones(x.shape[0])).to(self.device)
        # Use the model to predict the noise for the noisy data at the given time step
        pred_noise = self.model(x, t_repeat)
        # Calculate the value of the score function
        return self.st_derivative(t) / self.st(t) * x + self.st(t) * self.sigma_t_derivative(t) * pred_noise

    def heun_torch(self, xt, t, t_next):
        '''
        Integrates the data at a given time step using the Heun method to estimate the data at the next time step.

        The Heun method, an improved version of the Euler method, increases estimation accuracy by averaging the slopes at the current and next time steps.

        Parameters:
        - xt (Tensor): The data at the current time step t.
        - t (float): The current time step.
        - t_next (float): The next time step.

        Returns:
        - Tensor: The estimated data at time step t_next.
        '''
        # Calculate the slope at the current time step t
        dx = self.score_function(xt, t) * (t_next - t)
        # Estimate the data at the next time step using the current slope
        xt_tilde = xt + dx

        # Calculate the slope at the estimated next time step t_next
        dx_tilde = self.score_function(xt_tilde, t_next) * (t_next - t)

        # Update the data using the average of the current and estimated next slopes
        return xt + (dx + dx_tilde) / 2

    def exact_dynamics_heun(self, xT, timesteps): # do a lot of Heun steps between tn and tn1 to get the accurate xtn
        xt = xT
        for i in range(len(timesteps)-1):
            t = timesteps[i]
            tnext = timesteps[i+1]
            xt = self.heun_torch(xt, t, tnext)
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



    
    
