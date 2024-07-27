import torch
import numpy as np
import os
import psutil  # pip install psutil

def clean_up():
    """
    Terminates all child processes of the current process.

    Uses the psutil library to get an instance of the current process, then iterates over and terminates all child processes.
    Finally, prints a message confirming all relevant processes have been terminated.
    """
    current_process = psutil.Process(os.getpid())  # Get an instance of the current process
    for child in current_process.children(recursive=True):  # Iterate over all child processes
        child.terminate()  # Terminate the child process
    
    print("All relevant processes terminated")  # Print termination confirmation message

def remove_mean(samples, n_particles, n_dimensions):
    """
    Makes a configuration of many particle system mean-free.

    Parameters
    ----------
    samples : torch.Tensor
        Positions of n_particles in n_dimensions.

    Returns
    -------
    samples : torch.Tensor
        Mean-free positions of n_particles in n_dimensions.
    """

    shape = samples.shape
    if isinstance(samples, torch.Tensor):
        samples = samples.view(-1, n_particles, n_dimensions)
        samples = samples - torch.mean(samples, dim=1, keepdim=True)
        samples = samples.view(*shape)
    else:
        samples = samples.reshape(-1, n_particles, n_dimensions)
        samples = samples - samples.mean(axis=1, keepdims=True)
        samples = samples.reshape(*shape)
    return samples

def modify_samples_torch_batched_K(x, weights, mean=0.0, std=1.0, K=1):
    """
    For each sample in the tensor x, randomly pick K dimensions and change the coordinates of those dimensions
    to Gaussian random variables with the specified mean and standard deviation, using PyTorch,
    with batched operations instead of a loop.
    
    Parameters:
    - x: A PyTorch tensor of shape (nbatch, ndim).
    - mean: The mean of the Gaussian distribution.
    - std: The standard deviation of the Gaussian distribution.
    - K: The number of dimensions to modify.
    
    Returns:
    - A PyTorch tensor with modified samples.
    """
    nbatch, ndim = x.size()
    device = x.device
    #random_dims = torch.argsort(torch.rand(nbatch, ndim), dim=1)[:, :K]
    random_dims = torch.multinomial(weights, K)
    # print(random_dims)
    # Generate Gaussian random variables for each sample
    random_values = torch.normal(mean, std, (nbatch, K)).to(device)
    # Create a tensor of indices for batch indexing
    batch_indices = torch.arange(nbatch)
    # Modify the selected dimensions for each sample
    x[batch_indices[:, None], random_dims] = random_values
    return

def generate_tsampling(epsilon, tmax, Nselected, rho):
    """
    Generates a set of sampling times using a non-linear transformation to ensure more samples are
    taken close to the start time (t=0) and fewer as time approaches tmax. This can be useful in
    simulations where early time behavior is more complex or requires finer resolution.

    Parameters:
    - epsilon: A small positive value to avoid singularity at t=0.
    - tmax: The maximum time value.
    - Nselected: The number of time samples to generate.
    - rho: The exponent used in the non-linear transformation, controlling the distribution of samples.

    Returns:
    - tselected: An array of Nselected time samples.
    """
    # Generate Nselected time samples using a non-linear transformation
    tselected = (epsilon**(1.0/rho) + np.arange(Nselected) / (Nselected - 1) * (tmax**(1.0/rho) - epsilon**(1.0/rho)))**rho
    return tselected