from torch import nn
import torch
import math
import torch.nn.functional as F

class DiTBlock_new(nn.Module):
    def __init__(self, emb_size, nhead):
        super().__init__()
        self.emb_size = emb_size
        self.nhead = nhead

        # Merge six conditional linear layers into one, output 6 * emb_size, then split with chunk
        self.cond_linear = nn.Linear(emb_size, emb_size * 6)

        # Layer normalizations
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)

        # Merge wq, wk, wv into one qkv linear layer, output 3 * (nhead * emb_size)
        self.qkv = nn.Linear(emb_size, 3 * nhead * emb_size)
        self.lv = nn.Linear(nhead * emb_size, emb_size)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            nn.ReLU(),
            nn.Linear(emb_size * 4, emb_size)
        )

    def forward(self, x, cond):
        # Compute conditional linear transformation in parallel and split into six parts
        cond_vals = self.cond_linear(cond)  # (batch, 6*emb_size)
        gamma1_val, beta1_val, alpha1_val, gamma2_val, beta2_val, alpha2_val = cond_vals.chunk(6, dim=-1)

        # First layer normalization
        y = self.ln1(x)  # (batch, seq_len, emb_size)
        # Scale and shift using gamma1 and beta1
        y = y * (1 + gamma1_val.unsqueeze(1)) + beta1_val.unsqueeze(1)

        # Compute combined q, k, v
        qkv = self.qkv(y)  # (batch, seq_len, 3*nhead*emb_size)
        q, k, v = qkv.chunk(3, dim=-1)  # each is (batch, seq_len, nhead*emb_size)

        # Reshape q, k, v for multi-head attention
        q = q.view(q.size(0), q.size(1), self.nhead, self.emb_size).permute(0, 2, 1, 3)  # (batch, nhead, seq_len, emb_size)
        k = k.view(k.size(0), k.size(1), self.nhead, self.emb_size).permute(0, 2, 1, 3)  # (batch, nhead, seq_len, emb_size)
        v = v.view(v.size(0), v.size(1), self.nhead, self.emb_size).permute(0, 2, 1, 3)  # (batch, nhead, seq_len, emb_size)

        # Compute scaled dot-product attention
        # Alternative: use F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        #k = k.transpose(-2, -1)
        #attn = q @ k / math.sqrt(q.size(-1))  # (batch, nhead, seq_len, seq_len)
        #attn = torch.softmax(attn, dim=-1)    # (batch, nhead, seq_len, seq_len)
        #y = attn @ v                          # (batch, nhead, seq_len, emb_size)
        #y = y.permute(0, 2, 1, 3)             # (batch, seq_len, nhead, emb_size)
        #y = y.reshape(y.size(0), y.size(1), -1)  # (batch, seq_len, nhead*emb_size)
        y=F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)    # (batch,nhead,seq_len,emb_size)
        y = y.permute(0, 2, 1, 3).reshape(x.size(0), x.size(1), self.nhead * self.emb_size)
        y = self.lv(y)

        # First residual connection with scaling by alpha1
        y = y * alpha1_val.unsqueeze(1)
        y = x + y

        # Second residual branch
        z = self.ln2(y)
        z = z * (1 + gamma2_val.unsqueeze(1)) + beta2_val.unsqueeze(1)
        z = self.ff(z)
        z = z * alpha2_val.unsqueeze(1)
        return y + z

    
