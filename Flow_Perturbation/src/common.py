import torch
import torch.nn as nn
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
