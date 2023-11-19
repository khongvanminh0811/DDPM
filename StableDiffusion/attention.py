import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, dim_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(dim_embed, 3 * dim_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(dim_embed, dim_embed, bias=out_proj_bias)
        self.n_head = n_heads
        self.dim_head = dim_embed // n_heads

    def forward(self, x: torch.Tensor, casual_mask=False):
        input_shape = x.shape

        batch_size, sequence_length, dim_embed = input_shape

        intermim_shape = (batch_size, sequence_length, self.n_head, self.dim_head)

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (batch_size, sequence_length, dim_embed) -> (batch_size, sequence_length, dim_head, n_heads)
        #  (batch_size, sequence_length, dim_head, n_heads) -> (batch_size, dim_head, sequence_length, n_heads)
        q = q.view(intermim_shape).transpose(1,2)
        k = k.view(intermim_shape).transpose(1,2)
        v = v.view(intermim_shape).transpose(1,2)

        w = q @ k.transpose(-1, -2)
        if casual_mask:
            # masked upper triangle
            mask = torch.ones_like(w, dtype=torch.bool).triu(1)
            w.masked_fill(mask, -torch.inf)

        w /= math.sqrt(self.dim_head)
        w = F.softmax(w, dim=-1)
        output = w @ v
        output = output.reshape(input_shape)
        output = self.out_proj(output)

        return output

class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embd: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embd, d_embd, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embd, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embd, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embd, d_embd, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embd // n_heads


    def forward(self, x, y):
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        w = q @ k.transpose(-1, -2)
        w /= math.sqrt(self.d_head)
        w = F.softmax(w, dim=-1)

        output = w @ v
        output = output.transpose(1, 2).continuos()
        output = output.view(input_shape)
        output = self.out_proj(output)

        return output