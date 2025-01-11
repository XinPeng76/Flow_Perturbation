import torch
from .utils import generate_tsampling
from .odesolver import odesolver,odesolver_Huch_dSt

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
    @torch.no_grad()
    def ode_step(self, x, t, t_next):
        return odesolver(self.score_function_rearange, x, t, t_next)
    
    @torch.no_grad()
    def exact_dynamics(self, xT, timesteps): 
        xt = xT
        for i in range(len(timesteps)-1):
            t = timesteps[i]
            tnext = timesteps[i+1]
            xt = odesolver(self.score_function_rearange, xt, t, tnext)
        return xt
    
    def exact_dynamics_dSt(self, xT, timesteps, method = 'RK4',nnoise = 1, eps_type='Rademacher'): 
        xt = xT
        dSt = torch.zeros(xt.shape[0]).to(self.device)
        if nnoise >= 1:
            fun = lambda t, x: self.score_function_rearange(t, x)
        else:
            fun = lambda x, t: self.score_function_1element(x, t)
            
        for i in range(len(timesteps)-1):
            t = timesteps[i]
            tnext = timesteps[i+1]
            xt ,div_xt = odesolver_Huch_dSt(fun, xt, t, tnext, method, nnoise, eps_type)
            dSt += div_xt
        return xt, dSt

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

    @torch.no_grad()
    def ode_step(self, x, t, t_next):
        return odesolver(self.score_function_rearange, x, t, t_next)
    
    @torch.no_grad()
    def exact_dynamics(self, xT, timesteps): 
        xt = remove_mean(xT, self.n_particles, self.n_dimensions)
        for i in range(len(timesteps)-1):
            t = timesteps[i]
            tnext = timesteps[i+1]
            xt = odesolver(self.score_function_rearange, xt, t, tnext)
            xt = remove_mean(xt, self.n_particles, self.n_dimensions)
        return xt
    
    def exact_dynamics_dSt(self, xT, timesteps, method = 'RK4',nnoise = 1, eps_type='uniform'):
        xt = remove_mean(xT, self.n_particles, self.n_dimensions)
        dSt = torch.zeros(xt.shape[0]).to(self.device)
        if nnoise >= 1:
            fun = lambda t, x: self.score_function_rearange(t, x)
        else:
            fun = lambda x, t: self.score_function_1element(x, t)
            
        for i in range(len(timesteps)-1):
            t = timesteps[i]
            tnext = timesteps[i+1]
            xt ,div_xt = odesolver_Huch_dSt(fun, xt, t, tnext, method, nnoise, eps_type)
            dSt += div_xt
            xt = remove_mean(xt, self.n_particles, self.n_dimensions)
        return xt, dSt
