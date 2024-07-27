import torch
import numpy as np
from scipy.interpolate import CubicSpline

def calc_alphas_betas(num_steps=1000, scaling=10, beta_min=1e-5, beta_max=1e-2):
    '''
    Calculate the alpha and beta values for Denoising Diffusion Probabilistic Models (DDPM).

    Parameters:
    - num_steps (int): The number of diffusion steps.
    - scaling (float): A scaling factor to adjust the range of beta values.
    - beta_min (float): The minimum value of beta.
    - beta_max (float): The maximum value of beta.

    Returns:
    - alphas (torch.tensor): The alpha values for each step.
    - betas (torch.tensor): The beta values for each step.
    - alphas_prod (torch.tensor): The cumulative product of alpha values.
    - alphas_bar_sqrt (torch.tensor): The square root of the cumulative product of alpha values.
    - one_minus_alphas_bar_sqrt (torch.tensor): The square root of one minus the cumulative product of alpha values.
    '''
    # Generate linearly spaced beta values adjusted by the scaling factor and beta function.
    betas = torch.linspace(-scaling, scaling, num_steps)
    betas = torch.sigmoid(betas) * (beta_max - beta_min) + beta_min
    # Calculate alpha values as 1 minus beta values.
    alphas = 1 - betas
    # Calculate the cumulative product of alpha values.
    alphas_prod = torch.cumprod(alphas, 0)
    # Calculate the square root of the cumulative product of alpha values.
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    # Calculate the square root of one minus the cumulative product of alpha values.
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    return alphas, betas, alphas_prod, alphas_bar_sqrt, one_minus_alphas_bar_sqrt

def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    '''
    Compute the diffusion loss for a given model and input.

    Parameters:
    - model: The DDPM model to compute the loss for.
    - x_0 (torch.Tensor): The original input data.
    - alphas_bar_sqrt (torch.Tensor): The square root of the cumulative product of alpha values.
    - one_minus_alphas_bar_sqrt (torch.Tensor): The square root of one minus the cumulative product of alpha values.
    - n_steps (int): The number of diffusion steps.

    Returns:
    - The mean squared error between the noise predicted by the model and the actual noise.
    '''
    # Get the device and batch size from the input.
    device = x_0.device
    batch_size = x_0.shape[0]
    # Randomly select a timestep for each data in the batch.
    t = torch.randint(0, n_steps, size=(batch_size,))
    t = t.unsqueeze(-1)
    # Adjust the selected timesteps for the input and noise.
    a = alphas_bar_sqrt[t].to(device)
    aml = one_minus_alphas_bar_sqrt[t].to(device)
    # Generate random noise.
    e = torch.randn_like(x_0).to(device)
    # Create the noisy data by combining the original data with the noise.
    x = x_0 * a + e * aml
    # Predict the noise for the noisy data at the selected timesteps.
    t = t.to(device)
    output = model(x, t.squeeze(-1))
    # Calculate the mean squared error loss.
    return (e - output).square().mean()

def interpolate_parameters(num_steps, alphas_prod):
    '''
    Following the appendix section "flow model for Chignolin" from the paper,
    Interpolates the alpha product and its inverse ratio as continuous functions over time steps.

    Parameters:
    - num_steps (int): The number of diffusion steps.
    - alphas_prod (np.array): The cumulative product of alpha values.

    Returns:
    - st (CubicSpline): A cubic spline interpolation of the square root of alphas_prod.
    - sigma_t (CubicSpline): A cubic spline interpolation of the square root of the inverse ratio of alphas_prod.
    - st_derivative (CubicSpline): The derivative of st.
    - sigma_t_derivative (CubicSpline): The derivative of sigma_t.
    '''
    # Generate an array of time steps from 0 to num_steps-1
    time_steps = np.arange(num_steps)
    # Interpolate to obtain st and sigma_t as described in the paper
    st = CubicSpline(time_steps, np.sqrt(alphas_prod))
    sigma_t = CubicSpline(time_steps, np.sqrt((1-alphas_prod)/alphas_prod))
    # Calculate the derivative of st
    st_derivative = st.derivative()
    # Calculate the derivative of sigma_t
    sigma_t_derivative = sigma_t.derivative()

    return st, sigma_t, st_derivative, sigma_t_derivative

from .utils import remove_mean

class DDPMSamplerCoM:
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
    def score_function_rearange(self, t, x):
        return self.score_function(x, t)
    def score_function_1element(self, x, t):
        x = x.reshape(1,-1)
        score = self.score_function(x, t)
        return score.flatten()
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
    
    def score_function_rearange(self, t, x):
        return self.score_function(x, t)
    
    def score_function_1element(self, x, t):
        x = x.reshape(1,-1)
        score = self.score_function(x, t)
        return score.flatten()


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
