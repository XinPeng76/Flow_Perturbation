import torch
import numpy as np
import os
import psutil  # pip install psutil
import time
import logging
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

def modify_samples_torch_batched_K(x, mean=0.0, std=1.0, K=1):
    """
    For each sample in the tensor x, randomly pick K dimensions and change the coordinates of those dimensions
    to Gaussian random variables with the specified mean and standard deviation, using PyTorch,
    with batched operations instead of a loop.
    
    Parameters:
    - x: A PyTorch tensor of shape (nbatch, ndim).
    - mean: The mean of the Gaussian distribution.
    - std: The standard deviation of the Gaussian distribution.
    - K: The number of dimensions to modify or a tensor of shape (nbatch,) to specify the number of dimensions to modify for each sample.

    Returns:
    - A PyTorch tensor with modified samples.
    """
    nbatch, ndim = x.size()
    if isinstance(K, int):
        K = torch.ones(x.size(0), device=x.device, dtype=torch.long) * K
    # Generate random values to pick dimensions
    rand_vals = torch.rand(nbatch, ndim, device=x.device)  # Random values for each dimension
    sorted_indices = rand_vals.argsort(dim=1)  # Sort indices based on random values for each row

    # K_matrix expands K values for broadcasting
    K_matrix = K.unsqueeze(1).expand(-1, ndim)
    
    # Generate mask for top K selected dimensions per row
    random_dims_mask = sorted_indices < K_matrix  # Use sorted indices 

    # Random Gaussian values for modification
    random_values = torch.normal(mean, std, (nbatch, ndim), device=x.device)
    
    # Modify only the selected dimensions
    x[random_dims_mask] = random_values[random_dims_mask]

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

def generate_betasampling(epsilon, tmax, Nselected, rho):
    tselected = (epsilon**(1.0/rho) + torch.arange(Nselected, dtype=torch.float32) / (Nselected - 1) * (tmax**(1.0/rho) - epsilon**(1.0/rho)))**rho
    return tselected

def get_new_log_dir(root='./logs', prefix='', tag=''):
    fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    if prefix != '':
        fn = prefix + '_' + fn
    if tag != '':
        fn = fn + '_' + tag
    log_dir = os.path.join(root, fn)
    os.makedirs(log_dir)
    return log_dir
def get_logger(name, log_dir=None, log_fn='log.txt'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, log_fn))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def get_beta_schedule(num_steps, beta_min=-2, beta_max=1.0,beta_cut=0.0):
    #beta = torch.linspace(-2, 1, (num_steps))
    beta = torch.linspace(beta_min, beta_max, (num_steps))
    beta = torch.sigmoid(beta)
    beta = (beta - beta.min()) / (beta.max() - beta.min())
    shift = 1 - beta[-1]
    beta = beta + shift
    beta = beta * (1-beta_cut) + beta_cut
    return beta

def generate_K_values(n_steps, Kmax_x=20, Kmax_eps=20, Kmin_x=1, Kmin_eps=1, 
                      compress_exp_x=0.05, compress_exp_eps=0.05, device='cuda'):
    betak = generate_betasampling(0, 1, n_steps, 2).to(device)
    
    # Shift betak to ensure it starts from 1
    shift = 1 - betak[-1]
    betak = betak + shift
    
    # Compute K_x using the given parameters and formula
    K_x = (Kmin_x * betak**compress_exp_x + Kmax_x * (1 - betak**compress_exp_x)).int().to(device)
    
    # Compute K_eps using the given parameters and formula
    K_eps = (Kmin_eps * betak**compress_exp_eps + Kmax_eps * (1 - betak**compress_exp_eps)).int().to(device)
    
    return K_x, K_eps