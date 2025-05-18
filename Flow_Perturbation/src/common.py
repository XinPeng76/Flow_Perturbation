import torch
import torch.nn as nn
import numpy as np

class LangevinDynamicsWithLogP(torch.nn.Module):
    def __init__(self, eta, beta, potential_fn):
        super(LangevinDynamicsWithLogP, self).__init__()
        self.eta = eta  # Time step size ϵ_t, controls the step size of the dynamics
        self.beta = beta  # Inverse temperature, typically set to 1 for simplicity
        self.potential_fn = potential_fn  # Potential energy function u_lambda(y), which is used to compute the gradients
        
    def forward(self, y):
        # Compute the gradient of the potential energy function with respect to y
        grad_u = self.potential_fn(y)
        
        # Generate forward noise η_t ~ N(0, I), where the noise is drawn from a normal distribution
        # The shape of the noise is the same as y
        noise_forward = torch.randn_like(y)  # Noise with the same shape as y
        # Update y based on Langevin dynamics: y_next = y - ϵ_t * grad_u + sqrt(2 * ϵ_t / β) * η_t
        y_next = y - self.eta * grad_u + np.sqrt(2 * self.eta / self.beta) * noise_forward
        
        # Compute the gradient of the potential energy at the next position y_next
        grad_u_next = self.potential_fn(y_next)
    
        # Compute the backward noise term based on the Langevin dynamics equations
        # This is used for calculating the log path probability ratio
        noise_backward = np.sqrt(self.beta * self.eta / 2) * (grad_u + grad_u_next) - noise_forward
        
        # Calculate the change in the log-probability (log path probability ratio)
        delta_S_t = -0.5 * torch.sum(noise_backward**2 - noise_forward**2, dim=-1)
        
        return y_next, delta_S_t  # Return the updated position and the log path probability change