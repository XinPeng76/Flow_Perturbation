import torch
import torch.nn as nn
import numpy as np
from .positional_embeddings import PositionalEmbedding

class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(size)

    def forward(self, x: torch.Tensor):
        return self.norm(x + self.act(self.ff(x)))
        


class MLP(nn.Module):
    def __init__(self,ndim: int, hidden_size: int =  2048, hidden_layers: int = 12, emb_size: int = 10,
                 time_emb: str = "sinusoidal"):
        super().__init__()

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        concat_size = ndim + len(self.time_mlp.layer)
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, ndim))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = torch.cat((x, t_emb), dim=-1)
        x = self.joint_mlp(x)
        return x


class Block_nonorm(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.ff(x))

class MLP_nonorm(nn.Module):
    def __init__(self,ndim: int, hidden_size: int =  2048, hidden_layers: int = 12, emb_size: int = 10,
                 time_emb: str = "sinusoidal"):
        super().__init__()

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        concat_size = ndim + len(self.time_mlp.layer)
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block_nonorm(hidden_size))
        layers.append(nn.Linear(hidden_size, ndim))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = torch.cat((x, t_emb), dim=-1)
        x = self.joint_mlp(x)
        return x



class MLP_var(nn.Module):
    def __init__(self,ndim: int, hidden_size: int = 40, hidden_layers: int = 10):
        super().__init__()

        concat_size = ndim
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block_nonorm(hidden_size))
        layers.append(nn.Linear(hidden_size, 1))
        self.joint_mlp = nn.Sequential(*layers)
        self.scalar = nn.Parameter(0.01*torch.randn(1))
    def forward(self, x):
        x = torch.abs(self.scalar) + 0.01 * torch.abs(self.joint_mlp(x))+1e-6
        return x


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