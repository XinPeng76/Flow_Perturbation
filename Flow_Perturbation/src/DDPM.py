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
