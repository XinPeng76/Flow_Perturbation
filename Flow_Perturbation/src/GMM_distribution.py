import torch
import numpy as np

def random_rotation_matrix(n, rng):
    # Generate a random n x n matrix
    A = torch.normal(0, 1, (n, n), generator=rng)
    # Perform QR decomposition
    Q, R = torch.linalg.qr(A)
    # Ensure a proper rotation matrix (det(Q) should be 1)
    return Q

def generate_covariance_matrix(d, rng, diag):
    # Generate a random matrix using torch.normal, passing the generator
    sigmas = 0.01 + torch.abs(torch.normal(0.0, 0.25, size=(d,), generator=rng))
    if diag == True:
        return torch.diag(sigmas)
    Q = random_rotation_matrix(d, rng)
    return  Q @ torch.diag(sigmas) @ Q.T

def sample_NWell(nsamples, ndim, nwell):
    # Set manual seed for reproducibility in PyTorch    
    # Use PyTorch's random number generation
    rng = torch.Generator().manual_seed(42)
    
    # Generate location of wells using rng
    #mus = torch.normal(0, 1, size=(nwell, ndim), generator=rng)
    mus = torch.zeros(nwell, ndim)
    mus[0, 0] = -2.0
    mus[1, 0] = 2.0
    # Generate covariance matrix for each well
    sigmas = torch.zeros((nwell, ndim, ndim))
    #sigmas[0] = generate_covariance_matrix(ndim, rng, diag=True)
    #sigmas[1] = generate_covariance_matrix(ndim, rng, diag=False)
    #for i in range(nwell):
    sigmas_diag = 0.01 + torch.abs(torch.normal(0.0, 0.25, size=(ndim,), generator=rng))
    A = torch.normal(0, 1, (ndim, ndim), generator=rng)
    # Perform QR decomposition
    Q, R = torch.linalg.qr(A)
    sigmas[1] = Q @ torch.diag(sigmas_diag) @ Q.T

    sigmas_diag = 0.01 + torch.abs(torch.normal(0.0, 0.25, size=(ndim,), generator=rng))
    sigmas[0] = torch.diag(sigmas_diag)
        #sigmas[i] = (i+1) * generate_covariance_matrix(ndim, rng)
    # Generate the coefficients for each well
    coeffs = 3**torch.arange(nwell)
    #coeffs = torch.ones(nwell)
    coeffs = coeffs / torch.sum(coeffs)
    # Generate samples for each well using torch.distributions.MultivariateNormal
    samples = torch.zeros((nwell, nsamples, ndim))

    mvn_list = []
    for i in range(nwell):
        dist = torch.distributions.MultivariateNormal(mus[i], sigmas[i])
        samples[i] = dist.sample((nsamples,))  # Generate nsamples for each well
        mvn_list.append(dist)
        
    return samples, [mus, sigmas, coeffs], mvn_list

def redraw_samples(samples, Nwellinfo):
    coeff = Nwellinfo[-1]
    coeff_normalized = coeff / coeff.sum()
    total_samples = samples.shape[1]  # Example total number of samples
    samples_per_row = (coeff_normalized * total_samples).long()
    sampled_elements = []
    # Draw samples from each row
    for i in range(samples.shape[0]):
        indices = torch.randperm(samples.shape[1])[:samples_per_row[i]]
        sampled_elements.append(samples[i, indices])
    sampled_elements = torch.cat(sampled_elements)
    return sampled_elements

def sample_new_data(nsamples, ndim, nwell, Nwell_info):
    # sample new data based on the input Nwell_info
    mus = Nwell_info[0]
    sigmas = Nwell_info[1]
    coeffs = Nwell_info[2]

    samples = torch.zeros((nwell, nsamples, ndim), device=mus.device)  # 使用与 mus 相同的设备
    for i in range(nwell):
        dist = torch.distributions.MultivariateNormal(mus[i], sigmas[i])
        samples[i] = dist.sample((nsamples,))

    return samples.view(-1, ndim).to(mus.device)  # reshape to (-1, ndim)

def get_energy_device(x, mvn_list, Nwellinfo):
    mus = Nwellinfo[0]
    coeffs = Nwellinfo[2]
    logpx = torch.zeros(len(mvn_list), x.shape[0], device=x.device)
    for i in range(len(mvn_list)):
        logpx[i] = mvn_list[i].log_prob(x) + torch.log(coeffs[i])
    logpx_max, _ = torch.max(logpx, dim=0)
    delta_logpx = logpx - logpx_max
    logpx_max += torch.log(torch.sum(torch.exp(delta_logpx), dim=0))
    return -logpx_max


def log_prob_grad(x, mu, cov_matrix):
    """
    手动计算多元正态分布对数概率的梯度
    参数:
    x : torch.Tensor - 形状为 (N, D) 的样本点，N 是样本数，D 是维度数
    mu : torch.Tensor - 形状为 (D,) 的均值
    cov_matrix : torch.Tensor - 形状为 (D, D) 的协方差矩阵
    
    返回:
    grad : torch.Tensor - 形状为 (N, D) 的梯度
    """
    # 计算协方差矩阵的逆
    cov_inv = torch.inverse(cov_matrix)
    
    # 计算 x - mu
    diff = x - mu
    
    # 计算梯度
    grad = torch.matmul(diff, cov_inv)
    
    return grad

def get_energy_gradient_device(x, mvn_list, Nwellinfo):
    mus = Nwellinfo[0]
    sigmas = Nwellinfo[1]
    coeffs = Nwellinfo[2]
    
    # 计算每个分布的 log_prob 和 logpx
    logpx = torch.zeros(len(mvn_list), x.shape[0], device=x.device)
    for i in range(len(mvn_list)):
        logpx[i] = mvn_list[i].log_prob(x) + torch.log(coeffs[i])
    
    # 计算 logpx_max
    logpx_max, _ = torch.max(logpx, dim=0)
    
    # 计算 logpx - logpx_max
    delta_logpx = logpx - logpx_max
    
    logpx_max += torch.log(torch.sum(torch.exp(delta_logpx), dim=0))

    # 计算梯度
    grad = torch.zeros_like(x)
    for i in range(len(mvn_list)):
        mvn_grad = log_prob_grad(x, mus[i], sigmas[i])
        grad += torch.exp(logpx[i] - logpx_max).unsqueeze(-1) * mvn_grad
    return grad
