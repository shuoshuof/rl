from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from rl.utils import resolve_nn_activation



class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers,
        hidden_dim,
        num_heads,
        ff_dim=None,
        activation='silu',
        dropout=0.0,
        block_variant: str = "postnorm",
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_dim,
                num_heads,
                ff_dim,
                activation,
                dropout,
                block_variant=block_variant,
            )
            for _ in range(num_layers)
        ])

    def forward(self, src, attn_mask: Optional[torch.Tensor]=None, is_causal=False):
        for layer in self.layers:
            src = layer(src, attn_mask=attn_mask, is_causal=is_causal)
        return src



class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        ff_dim=None,
        activation='silu',
        dropout=0.0,
        block_variant: str = "postnorm",
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.ff = FeedForward(hidden_dim, ff_dim, activation, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.block_variant = block_variant

    def forward(self, src, attn_mask: Optional[torch.Tensor]=None, is_causal=False):
        if self.block_variant == "prenorm":
            q = self.norm1(src)
            self_attn_output = self.self_attn(q, q, q, attn_mask=attn_mask, is_causal=is_causal)
            self_attn_output = self.dropout1(self_attn_output)
            src = src + self_attn_output

            src_norm = self.norm2(src)
            ff_out = self.ff(src_norm)
            ff_out = self.dropout2(ff_out)
            src = src + ff_out
        elif self.block_variant == "postnorm":
            self_attn_output = self.self_attn(src, src, src, attn_mask=attn_mask, is_causal=is_causal)
            self_attn_output = self.dropout1(self_attn_output)
            src = self.norm1(src + self_attn_output)

            ff_out = self.ff(src)
            ff_out = self.dropout2(ff_out)
            src = self.norm2(src + ff_out)
        else:
            raise ValueError(f"Unsupported block variant: {self.block_variant}")
        return src

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers,
        hidden_dim,
        num_heads,
        ff_dim=None,
        activation='silu',
        dropout=0.0,
        block_variant: str = "postnorm",
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                hidden_dim,
                num_heads,
                ff_dim,
                activation,
                dropout,
                block_variant=block_variant,
            )
            for _ in range(num_layers)
        ])

    def forward(self, q, k, v):
        for layer in self.layers:
            q = layer(q, k, v)
        return q


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        ff_dim=None,
        activation='silu',
        dropout=0.0,
        block_variant: str = "postnorm",
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.cross_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ff = FeedForward(hidden_dim, ff_dim, activation, dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.block_variant = block_variant

    def forward(self, q, k, v):
        if self.block_variant == "prenorm":
            self_q = self.norm1(q)
            self_attn_output = self.self_attn(self_q, self_q, self_q)
            self_attn_output = self.dropout1(self_attn_output)
            q = q + self_attn_output

            cross_q = self.norm2(q)
            cross_attn_output = self.cross_attn(cross_q, k, v)
            cross_attn_output = self.dropout2(cross_attn_output)
            q = q + cross_attn_output

            q_norm = self.norm3(q)
            ff_out = self.ff(q_norm)
            ff_out = self.dropout3(ff_out)
            q = q + ff_out
        elif self.block_variant == "postnorm":
            self_attn_output = self.self_attn(q, q, q)
            self_attn_output = self.dropout1(self_attn_output)
            q = self.norm1(q + self_attn_output)

            cross_attn_output = self.cross_attn(q, k, v)
            cross_attn_output = self.dropout2(cross_attn_output)
            q = self.norm2(q + cross_attn_output)

            ff_out = self.ff(q)
            ff_out = self.dropout3(ff_out)
            q = self.norm3(q + ff_out)
        else:
            raise ValueError(f"Unsupported block variant: {self.block_variant}")
        return q



class FeedForward(nn.Module):
    def __init__(self, hidden_dim, ff_dim=None, activation='silu', dropout=0.0):
        super().__init__()
        # TODO: bias = False, activation fixed to silu?
        ff_dim = ff_dim or hidden_dim * 4
        self.linear1 = nn.Linear(hidden_dim, ff_dim)
        self.linear2 = nn.Linear(hidden_dim, ff_dim)
        self.linear3 = nn.Linear(ff_dim, hidden_dim)
        self.activation = resolve_nn_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = x * gate
        x = self.dropout(x)
        return self.linear3(x)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_rate = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, attn_mask: Optional[torch.Tensor]=None, is_causal=False):
        B, Tq, C = q.shape
        B, Tk, C = k.shape
        q = self.q_proj(q).view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)
        # TODO: check whether can use torch's built-in scaled_dot_product_attention for better performance and stability, but need to verify the attn_mask format and causal mask behavior first.
        # q, k, v shape: (B, num_heads, T, head_dim). Use einsum here to keep
        # Inductor from rewriting this path to flash attention kernels that are
        # unstable for the MTBTv3 17-token body-attention shape in Isaac Sim's
        # bundled torch build.
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # attn_weights = torch.einsum("bhqd,bhkd->bhqk", q, k) / (self.head_dim ** 0.5)
        # attn_weights shape: (B, num_heads, Tq, Tk)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # attn_output = torch.einsum("bhqk,bhkd->bhqd", attn_weights, v)
        attn_output = torch.einsum("bhqk,bhkd->bhqd", attn_weights, v)

        # attn_output = F.scaled_dot_product_attention(
        #     q, k, v,
        #     attn_mask=attn_mask,
        #     dropout_p=self.dropout_rate if self.training else 0.0,
        #     is_causal=is_causal,
        # )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Tq, C)
        return self.proj_dropout(self.out_proj(attn_output))
    

if __name__ == "__main__":
    batch_size = 2
    seq_length = 10
    embed_dim = 64
    num_heads = 8
    num_layers = 3

    q = torch.randn(batch_size, seq_length+2, embed_dim)
    k = torch.randn(batch_size, seq_length, embed_dim)
    v = torch.randn(batch_size, seq_length, embed_dim)

    mha = MultiHeadAttention(embed_dim, num_heads, dropout=0.)
    mha_output = mha(q, k, v)
