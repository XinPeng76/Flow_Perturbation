from torch import nn
import torch
import torch.nn.functional as F
from .time_emb import TimeEmbedding
from .dit_block import DiTBlock_new

class DiT(nn.Module):
    def __init__(self, img_size, patch_size, channel, emb_size, dit_num, head):
        super().__init__()
        
        self.patch_size = patch_size
        # Compute adjusted image size to be the smallest multiple of patch_size
        self.adjusted_img_size = ((img_size + patch_size - 1) // patch_size) * patch_size
        # Number of patches per dimension
        self.patch_count = self.adjusted_img_size // patch_size
        self.channel = channel
        self.sqrt_ndim = img_size
        # Convolutional layer for patching
        self.conv = nn.Conv2d(
            in_channels=channel,
            out_channels=channel * patch_size ** 2,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )
        # Linear layer to embed patches
        self.patch_emb = nn.Linear(channel * patch_size ** 2, emb_size)
        # Positional embeddings based on adjusted image size
        self.patch_pos_emb = nn.Parameter(torch.rand(1, self.patch_count ** 2, emb_size))
        
        # Time embedding network
        self.time_emb = nn.Sequential(
            TimeEmbedding(emb_size),
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size)
        )

        # List of DiT blocks
        self.dits = nn.ModuleList([
           DiTBlock_new(emb_size, head) for _ in range(dit_num)
        ])
        
        # Layer normalization
        self.ln = nn.LayerNorm(emb_size)
        
        # Linear layer to project back to pixel space
        self.linear = nn.Linear(emb_size, channel * patch_size ** 2)
        
    def forward(self, x, t):
        batch, ndim = x.shape
        h = self.sqrt_ndim
        w = self.sqrt_ndim
        # Compute padding width
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        original_size = (h, w)  # Save original dimensions
        # Pad and reshape input to square image
        x = F.pad(x, (0, (self.sqrt_ndim + pad_w) ** 2 - ndim))
        x = x.reshape(batch, 1, self.sqrt_ndim + pad_w, self.sqrt_ndim + pad_w)
        
        # Create patches via convolution
        x = self.conv(x)  # (batch, new_channel, patch_count, patch_count)
        # Rearrange dimensions to (batch, patch_count, patch_count, new_channel)
        x = x.permute(0, 2, 3, 1)
        # Flatten patches to sequence (batch, patch_count**2, new_channel)
        x = x.view(x.size(0), self.patch_count ** 2, x.size(3))
        
        # Patch embedding and add positional encoding
        x = self.patch_emb(x)  # (batch, patch_count**2, emb_size)
        x = x + self.patch_pos_emb  # (batch, patch_count**2, emb_size)
        
        # Compute time embeddings
        t_emb = self.time_emb(t)
        
        # Pass through DiT blocks
        for dit in self.dits:
            x = dit(x, t_emb)
        
        # Final layer normalization
        x = self.ln(x)  # (batch, patch_count**2, emb_size)
        
        # Project back to pixel space
        x = self.linear(x)  # (batch, patch_count**2, channel * patch_size**2)
        
        # Reconstruct image from patches
        x = x.view(
            batch, self.patch_count, self.patch_count,
            self.channel, self.patch_size, self.patch_size
        )  # (batch, patch_count, patch_count, channel, patch_size, patch_size)
        # Rearrange to (batch, channel, height, width)
        x = x.permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(
            batch, self.channel,
            self.patch_count * self.patch_size,
            self.patch_count * self.patch_size
        )
        
        # Crop to original image size
        x = x[:, :, :original_size[0], :original_size[1]]
        x = x.reshape(batch, self.sqrt_ndim ** 2)[:, :ndim]
        return x