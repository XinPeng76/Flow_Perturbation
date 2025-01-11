from .scheme import Euler, RK2, RK4, Heun, Euler_dSt, RK2_dSt, RK4_dSt, Heun_dSt
import torch

def odesolver(func, z, t, t_next, method = 'RK4'):
    if (method == 'Euler'):
        z_next = Euler().step(func, t, t_next - t, z)
    elif (method == 'RK2'):
        z_next = RK2().step(func, t, t_next - t, z)
    elif (method == 'RK4'):
        z_next = RK4().step(func, t, t_next - t, z)
    elif (method == 'Heun'):
        z_next = Heun().step(func, t, t_next - t, z)
    else:
        print('error unsupported method passed')
        return
    return z_next

from torch.func import vjp, vmap, jacrev

from functools import partial

def get_vjp_score(func, t, xt, eps):
    score, vjp_score = vjp(partial(func, t), xt)
    x_grad, = vjp_score(eps)
    return score, torch.sum(eps * x_grad, dim=-1)

def get_vjp_score_mnoise(func, t, xt, eps):
    score, vjp_score = vjp(partial(func, t), xt)
    x_grad, = vmap(vjp_score)(eps)
    return score, torch.mean(torch.sum(eps * x_grad, dim=-1),dim=0)

def get_jacobian_score(func, t, xt, eps=None):
    with torch.no_grad():
        score = func(xt, t)
    score = score.reshape(xt.shape[0], -1)
    jacobian_score = jacrev(func)
    v_jacobian_score = vmap(jacobian_score, in_dims=(0, None))
    jj_score_xt_new = v_jacobian_score(xt, t)
    div_xt_new = torch.einsum("...ii", jj_score_xt_new)
    return score, div_xt_new

def get_jacobian_score_batch(func, t, xt, eps=None, batch_size=100):
    # Use torch.no_grad to prevent gradient calculation during evaluation
    with torch.no_grad():
        score = func(xt, t)
    
    # Reshape the score to be a 2D tensor (batch_size, flattened_dim)
    score = score.reshape(xt.shape[0], -1)
    
    # Define the Jacobian function and vectorize it using vmap
    jacobian_score = jacrev(func)
    v_jacobian_score = vmap(jacobian_score, in_dims=(0, None))
    
    # Get the number of samples in xt (the batch size)
    n = xt.shape[0]
    jacobian_scores = []
    
    # Iterate over the data in batches
    for i in range(0, n, batch_size):
        # Select the batch from xt, the last batch may have fewer elements
        batch_xt = xt[i:i + batch_size]
        
        # Compute the Jacobian for the current batch
        jj_score_xt_new = v_jacobian_score(batch_xt, t)
        
        # Append the Jacobian result for the batch
        jacobian_scores.append(jj_score_xt_new)
    
    # Concatenate the Jacobian results from all batches
    jj_score_xt_new = torch.cat(jacobian_scores, dim=0)
    
    # Compute the trace by summing the diagonal elements of the Jacobian matrices
    div_xt_new = torch.einsum("...ii", jj_score_xt_new)
    
    return score, div_xt_new

def odesolver_Huch_dSt(score_func, xt, t, t_next, method = 'RK4',nnoise = 1, eps_type='Rademacher'):
    if nnoise == 1:
        func = lambda t, xt, eps: get_vjp_score(score_func, t, xt, eps)
        if eps_type == 'Rademacher':
            eps = torch.randint(0, 2, xt.shape, device=xt.device, dtype=xt.dtype) * 2 - 1
        elif eps_type == 'Gaussian':
            eps = torch.randn_like(xt, device=xt.device, dtype=xt.dtype)
        else:
            ValueError('unsupported eps type')
    elif nnoise > 1:
        func = lambda t, xt, eps: get_vjp_score_mnoise(score_func, t, xt, eps)
        if eps_type == 'Rademacher':
            eps = (torch.randint(0, 2, (nnoise, xt.shape[0], xt.shape[1]), device=xt.device, dtype=xt.dtype) * 2 - 1)
        elif eps_type == 'Gaussian':
            eps = torch.randn((nnoise, xt.shape[0], xt.shape[1]), device=xt.device, dtype=xt.dtype)
        else:
            ValueError('unsupported eps type')
    else:
        ndim = xt.shape[-1]
        if ndim >=100:
            func = lambda t, xt, eps: get_jacobian_score_batch(score_func, t, xt, eps)
        else:
            func = lambda t, xt, eps: get_jacobian_score(score_func, t, xt, eps)
        eps = None
    if (method == 'Euler'):
        z_next , div_z_next = Euler_dSt().step(func, t, t_next - t, xt, eps)
    elif (method == 'RK2'):
        z_next , div_z_next = RK2_dSt().step(func, t, t_next - t, xt, eps)
    elif (method == 'RK4'):
        z_next , div_z_next = RK4_dSt().step(func, t, t_next - t, xt, eps)
    elif (method == 'Heun'):
        z_next , div_z_next = Heun_dSt().step(func, t, t_next - t, xt, eps)
    else:
        print('error unsupported method passed')
        return
    return z_next, div_z_next
