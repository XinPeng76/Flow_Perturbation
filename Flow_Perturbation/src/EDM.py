import torch
from .utils import generate_tsampling

def generate_sig(mean, std_dev, batchsize):
    normal_dist = torch.normal(mean, std_dev, size=(batchsize,))
    log_normal_dist = torch.exp(normal_dist)
    return log_normal_dist

def cskip(sig, sig_data): # the coeff of x in the D(x, t) expression
    return sig_data**2/(sig**2 + sig_data**2)

def cout(sig, sig_data): # the coeff that scales F(x, t)
    return sig * sig_data / torch.sqrt(sig**2 + sig_data**2)

def cin(sig, sig_data): # the term that scales x inside F(x, t)
    return 1/torch.sqrt(sig**2 + sig_data**2)

def cnoise(sig): # the terms that scales the t term in F(x,t)
    return 1/4*torch.log(sig)

def loss_EDM(model, sig_data,device, batch_x):
    batch_size = batch_x.shape[0]
    y = batch_x
    sig = generate_sig(-1.2, 1.2, batch_size).unsqueeze(-1).to(device) # shape is (batch_size, 1)
    e = torch.randn_like(y).to(device) # shape is (batch_size, ndim)
    n = sig * e # scale the variance
    input1_F = cin(sig, sig_data) * (y + n)
    input2_F = cnoise(sig).squeeze(-1) # shape is (batch_size,)

    pred_F = model(input1_F, input2_F)

    effective_target = 1.0/cout(sig, sig_data)*(y - cskip(sig, sig_data)*(y + n))

    loss = (pred_F - effective_target).square().mean()
    return loss

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
    def score_function_rearange(self, sig, x):
        return self.score_function(x, sig)
    def score_function_1element(self, x, sig):
        x = x.reshape(1,-1)
        score = self.score_function(x, sig)
        return score.flatten()

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

from .utils import remove_mean
class EMDSamplerCoM:
    def __init__(self, model, sig_data, n_particles, n_dimensions, device):
        self.model = model
        self.sig_data = sig_data
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
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
        pred_F = remove_mean(pred_F, self.n_particles, self.n_dimensions)
        return cskip(sig, sig_data) * x + cout(sig, sig_data) * pred_F
    def score_function(self, x, sig):
        return (self.ideal_denoiser(x, self.sig_data, sig) - x)/sig**2
    def score_function_rearange(self, sig, x):
        return self.score_function(x, sig)
    def score_function_1element(self, x, sig):
        x = x.reshape(1,-1)
        score = self.score_function(x, sig)
        return score.flatten()

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
        xt = remove_mean(xtn1, self.n_particles, self.n_dimensions)
        for i in range(len(ts) - 1, 0, -1):
            #print(ts[i-1], ts[i])
            xt = self.heun_torch(ts[i-1], ts[i], xt)
            xt = remove_mean(xt, self.n_particles, self.n_dimensions)
            #print(xt)
        return xt
