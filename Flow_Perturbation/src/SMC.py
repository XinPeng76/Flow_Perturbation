import torch
import numpy as np
from .utils import remove_mean, modify_samples_torch_batched_K
import math
import heapq

def tv_reshuffle(weights):
    """
    Perform total‐variation resampling: map a probability vector `weights` of length S
    to a list of indices (of total length S) by rounding fractional allocations.

    Args:
        weights (list of float): normalized weights summing to 1.

    Returns:
        indices (list of int): resampled indices, length = S.
    """
    S = len(weights)
    # Scale weights by S to get real‐valued counts
    a = [w * S for w in weights]
    # Compute integer parts (floors) and fractional remainders
    floors = [math.floor(x) for x in a]
    deltas = [x - f for x, f in zip(a, floors)]
    # Decide how many extra counts to assign based on sum of remainders
    alpha = int(round(sum(deltas)))
    # Sort indices by descending remainder and take top alpha to increment
    idx_sorted = sorted(range(S), key=lambda i: deltas[i], reverse=True)
    g = floors.copy()
    for i in idx_sorted[:alpha]:
        g[i] += 1
    # Build the output index list by repeating each symbol s, g[s] times
    indices = []
    for s, count in enumerate(g):
        indices += [s] * count
    return indices

def kl_reshuffle(w):
    """
    Perform KL‐optimal resampling: allocate S samples to indices 0..S-1
    by maximizing the expected log‐weight gain at each step.

    Args:
        w (array‐like of float): unnormalized weights, length S.

    Returns:
        indices (ndarray of int): resampled indices, length = S.
    """
    S = len(w)
    # Min‐heap storing tuples (-gain, index)
    heap = []
    # Allocation counts for each index
    a = np.zeros(S, dtype=int)

    # Initial gain for assigning the first copy: log(w_s / 1)
    for s, ws in enumerate(w):
        gain = np.log(ws)  # f(1,s) - f(0,s)
        heapq.heappush(heap, (-gain, s))

    total = 0
    # Greedily assign one by one until total = S
    while total < S:
        neg_gain, s = heapq.heappop(heap)
        # increase the copy count for index s
        a[s] += 1
        total += 1
        # compute gain for next copy: 
        # C⁺(a,s) = (a+1)·log(w_s/(a+1)) - a·log(w_s/a)
        a_s = a[s]
        gain = (a_s + 1) * np.log(w[s] / (a_s + 1)) - a_s * np.log(w[s] / a_s)
        heapq.heappush(heap, (-gain, s))

    # Build the index array by repeating each index s, a[s] times
    indices = np.repeat(np.arange(S), a)
    return indices


def systematic_resampling(weights):
    """
    Perform systematic resampling given particle weights.

    Args:
        weights (np.ndarray): 1D array of normalized weights (should sum to 1).

    Returns:
        np.ndarray: Indices of the resampled particles, length = N.
    """
    weights = np.asarray(weights)
    N = weights.shape[0]

    # Step 1: Compute the cumulative sum (CDF) of the weights
    cdf = np.cumsum(weights)
    # Ensure the last value of the CDF is exactly 1 (avoids precision issues)
    cdf[-1] = 1.0

    # Step 2: Generate a single random offset
    start = np.random.rand() / N

    # Step 3: Generate positions using the systematic pattern
    positions = start + np.arange(N) / N

    # Step 4: Find the indices where positions fall in the CDF
    indices = np.searchsorted(cdf, positions, side='right')

    return indices

def generate_doubling_intervals_exclude_start(a, N):
    r = (1 - a) / (2**N)
    indices = torch.arange(1, N + 1)  # Start from 1 to exclude `a`
    sequence = a + r * (2**indices)
    return sequence

def find_closest_larger_element_desc(sequence, b):
    # Find elements larger than b
    larger_elements = sequence[sequence > b]
    if len(larger_elements) == 0:
        raise ValueError("No element in the sequence is larger than b.")
    # In a decreasing sequence, the last element in the filtered list is the closest
    closest_value = larger_elements[-1]
    closest_index = torch.where(sequence == closest_value)[0].item()
    return closest_value.item(), closest_index
    

def mc_step(xT, eps, log_omega, x0, ux, K_x, K_eps,get_log_omega, beta=1.0, tmax=1.0, nmc=2,if_K_eps=True,if_com = False,n_particles = 1, n_dimensions = 1):
    """
    Perform a single Monte Carlo (MC) step on the given parameters.

    Args:
        xT (torch.Tensor): Current state of xT, shape (n_samples, ndim).
        eps (torch.Tensor): Current state of eps, shape (n_samples, ndim).
        log_omega (torch.Tensor): Current log-omega values.
        x0 (torch.Tensor): Current x0 values.
        ux (torch.Tensor): Current ux values.
        K_x (int): Number of random dimensions to be modified for xT.
        K_eps (int): Number of random dimensions to be modified for eps.
        beta (float): Inverse temperature (often 1.0).
        tmax (float): Maximum standard deviation for the modification of xT.
        device (torch.device): The device (CPU or GPU) where tensors are located.
        nmc (int): Number of Monte Carlo steps.
        if_K_eps (bool): Whether to modify eps or not.
        if_com (bool): Whether to remove the mean or not.
        n_particles (int): Number of particles.
        n_dimensions (int): Number of dimensions.

    Returns:
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float:
        Updated xT, eps, log_omega, x0, ux, and acceptance rate.
    """
    accept_rate = 0.0

    for j in range(nmc):
        # Clone current values to preserve the original ones
        xT_new = xT.clone()
        eps_new = eps.clone()
        
        # Modify samples using modify_samples_torch_batched_K
        modify_samples_torch_batched_K(xT_new, mean=0.0, std=tmax, K=K_x)
        if if_K_eps:
            modify_samples_torch_batched_K(eps_new, mean=0.0, std=1.0, K=K_eps)
        if if_com:
            xT_new = remove_mean(xT_new, n_particles, n_dimensions)
            eps_new = remove_mean(eps_new, n_particles, n_dimensions)
        # Compute log-omega, x0, and ux using the new values
        log_omega_new, x0_new, ux_new = get_log_omega(xT_new, eps_new)
        
        # Detailed balance move factor, scaled by beta
        db_factor = torch.exp(beta * (log_omega_new - log_omega))  # move factor
        
        # Random probabilities for the acceptance step
        p = torch.rand(db_factor.shape[0]).to(xT.device)
    
        # Determine which samples will be moved
        index_move = p < db_factor  # The index to be moved
    
        # Update the values for the samples that are accepted
        xT[index_move] = xT_new[index_move]
        if len(eps_new.shape) == 3:
            eps[:,index_move] = eps_new[:,index_move]
        else:
            eps[index_move] = eps_new[index_move]
        log_omega[index_move] = log_omega_new[index_move]
        x0[index_move] = x0_new[index_move]
        ux[index_move] = ux_new[index_move]

        # Compute the acceptance rate
        accept_rate += torch.mean(index_move.float()) / nmc

    return xT, eps, log_omega, x0, ux, accept_rate

def resample_if_needed(ess, n_replicas, i, n_steps, xT, eps, log_omega, x0, ux, ansestors, total_logweight, weights,ess_threshold=0.95,resample_method=systematic_resampling):
    """
    Resample the particles if the effective sample size (ESS) is below a threshold or on the second-to-last step.

    Args:
        ess (float): Effective sample size.
        n_replicas (int): Number of replicas (samples).
        i (int): Current step in the process.
        n_steps (int): Total number of steps.
        xT (torch.Tensor): The current xT state of all replicas.
        eps (torch.Tensor): The current eps state of all replicas.
        log_omega (torch.Tensor): The current log_omega values.
        x0 (torch.Tensor): The current x0 values.
        ux (torch.Tensor): The current ux values.
        ansestors (torch.Tensor): The ancestors information used for resampling.
        weights (torch.Tensor): The weights used for resampling.
        ess_threshold (float): The threshold for the effective sample size.
        resample_method (callable): The resampling method to use (default is systematic_resampling).

    Returns:
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict:
        The resampled xT, eps, log_omega, x0, ux, ansestors
    """
    if (ess < n_replicas * ess_threshold) or (i == n_steps - 2):  # Criteria for resampling
        # Resample the particles
        resampled_indices = resample_method(weights.cpu().numpy())  # Get resampled indices
        xT = xT[resampled_indices]  # Get the resampled xT, eps, log_omega, x0, and ux
        if len(eps.shape) == 3:
            eps = eps[:, resampled_indices]
        else:
            eps = eps[resampled_indices]
        log_omega = log_omega[resampled_indices]
        x0 = x0[resampled_indices]
        ux = ux[resampled_indices]
        ansestors = ansestors[resampled_indices]
        print("Step ", i, " - Resampling performed")

        total_logweight = torch.zeros_like(log_omega)  # Reassign total_logweight to zero
        unique_elements, counts = torch.unique(ansestors, return_counts=True)
        
        print("Number of unique elements:", unique_elements.shape)


    return xT, eps, log_omega, x0, ux, ansestors, total_logweight


def dists5_ratio(x0, n_particles=175, n_dimensions=3):
    if isinstance(x0, torch.Tensor):
        x0 = x0.cpu().detach().numpy()
    x0 = x0.reshape(-1, n_particles, n_dimensions)
    coord_7CA = x0[:, 105, :]
    coord_10CA = x0[:, 150, :]
    dists5 = np.sqrt(np.sum((coord_7CA - coord_10CA)**2, axis=-1))
    ratio = np.sum(dists5 > 0.75) / len(dists5)
    return ratio

def x0_ratio(x0, threshold=0.0):
    return (x0[:, 0] < threshold).sum().item() / x0.shape[0]