import torch
from src.utils.common import remove_mean

# 定义一个类，其输入是model,st,st_derivative,sigma_t_derivative,device
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
    
# 定义一个类，其输入是model,st,st_derivative,sigma_t_derivative,device
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

    
    
