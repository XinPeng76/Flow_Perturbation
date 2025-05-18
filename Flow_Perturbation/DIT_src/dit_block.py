from torch import nn 
import torch 
import math 
import torch.nn.functional as F
class DiTBlock_new(nn.Module):
    def __init__(self, emb_size, nhead):
        super().__init__()
        self.emb_size = emb_size
        self.nhead = nhead

        # 将6个条件线性层合并成一个线性层，输出6*emb_size，再用chunk分割
        self.cond_linear = nn.Linear(emb_size, emb_size * 6)

        # layer norm
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)

        # 将wq、wk、wv合并为一个qkv线性层，输出3 * (nhead*emb_size)
        self.qkv = nn.Linear(emb_size, 3 * nhead * emb_size)
        self.lv = nn.Linear(nhead * emb_size, emb_size)

        # feed-forward
        self.ff = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            nn.ReLU(),
            nn.Linear(emb_size * 4, emb_size)
        )

    def forward(self, x, cond):
        # 并行计算条件的线性变换，并分割为6个变量
        cond_vals = self.cond_linear(cond)  # (batch, 6*emb_size)
        gamma1_val, beta1_val, alpha1_val, gamma2_val, beta2_val, alpha2_val = cond_vals.chunk(6, dim=-1)

        # layer norm
        y = self.ln1(x)  # (batch, seq_len, emb_size)
        # scale & shift
        y = y * (1 + gamma1_val.unsqueeze(1)) + beta1_val.unsqueeze(1)

        # 合并计算q, k, v
        qkv = self.qkv(y)  # (batch, seq_len, 3*nhead*emb_size)
        q, k, v = qkv.chunk(3, dim=-1)  # 每个为 (batch, seq_len, nhead*emb_size)

        # 将q、k、v reshape为多头格式
        q = q.view(q.size(0), q.size(1), self.nhead, self.emb_size).permute(0, 2, 1, 3)  # (batch, nhead, seq_len, emb_size)
        k = k.view(k.size(0), k.size(1), self.nhead, self.emb_size).permute(0, 2, 1, 3)  # (batch, nhead, emb_size, seq_len)
        v = v.view(v.size(0), v.size(1), self.nhead, self.emb_size).permute(0, 2, 1, 3)  # (batch, nhead, seq_len, emb_size)
        
        # 计算注意力（缩放因子使用q的最后一维）
        #y=F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)    # (batch,nhead,seq_len,emb_size)
        k= k.transpose(-2, -1)
        attn=q@k/math.sqrt(q.size(2))   # (batch,nhead,seq_len,seq_len)
        attn=torch.softmax(attn,dim=-1)   # (batch,nhead,seq_len,seq_len)
        y=attn@v    # (batch,nhead,seq_len,emb_size)
        y=y.permute(0,2,1,3) # (batch,seq_len,nhead,emb_size)
        y=y.reshape(y.size(0),y.size(1),y.size(2)*y.size(3))    # (batch,seq_len,nhead*emb_size)
        y=self.lv(y)

        # 第一次scale和残差
        y = y * alpha1_val.unsqueeze(1)
        y = x + y

        # 第二次残差分支
        z = self.ln2(y)
        z = z * (1 + gamma2_val.unsqueeze(1)) + beta2_val.unsqueeze(1)
        z = self.ff(z)
        z = z * alpha2_val.unsqueeze(1)
        return y + z
    