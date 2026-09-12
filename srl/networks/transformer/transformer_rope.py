from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from srl.networks.transformer.transformer import FeedForward
from srl.utils import resolve_nn_activation



class RoPETransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers,
        hidden_dim,
        num_heads,
        ff_dim=None,
        activation='silu',
        dropout=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(
                hidden_dim,
                num_heads,
                ff_dim,
                activation,
                dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src




class RoPETransformerEncoderLayer(nn.Module):

    def __init__(
            self, 
            hidden_dim, 
            num_heads, 
            ff_dim=None, 
            activation='silu', 
            dropout=0.0, 
        ):

        super().__init__()
        self.attn = RoPEMultiHeadAttention(hidden_dim, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)


        self.ff = FeedForward(hidden_dim, ff_dim, activation, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.activation1 = resolve_nn_activation(activation)
        self.activation2 = resolve_nn_activation(activation)

    def forward(self, src: torch.Tensor) -> torch.Tensor:

        q = self.norm1(src)

        attn_output = self.attn(q, q, q)
        attn_output = self.activation1(attn_output)
        attn_output = self.dropout1(attn_output)
        src = src + attn_output

        src_norm = self.norm2(src)
        ff_output = self.ff(src_norm)
        ff_output = self.dropout2(self.activation2(ff_output))
        src = src + ff_output

        return src


class RoPEMultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.rotary_emb = RoPEEmbedding(
            head_dim=self.head_dim,
            base=rope_base,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ):
        """
        q: [B, Tq, C]
        k: [B, Tk, C]
        v: [B, Tk, C]
        """
        B, Tq, C = q.shape
        Bk, Tk, Ck = k.shape

        assert B == Bk
        assert C == self.embed_dim
        assert Ck == self.embed_dim

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # [B, T, C] -> [B, H, T, D]
        q = q.view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(
            seq_len=max(Tq, Tk),
            device=q.device,
            dtype=q.dtype,
        )

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / (self.head_dim ** 0.5)

        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)

        # [B, H, Tq, D] -> [B, Tq, C]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Tq, C)

        return self.proj_dropout(self.out_proj(attn_output))
    

class RoPEEmbedding(nn.Module):

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()

        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.head_dim = head_dim
        self.base = base

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        positions = torch.arange(
            seq_len,
            device=device,
            dtype=self.inv_freq.dtype,
        )

        # freqs: [seq_len, head_dim // 2]
        # freqs[m, i] = m * theta_i
        freqs = torch.einsum("t,d->td", positions, self.inv_freq.to(device))

        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)

        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)

        return cos, sin

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    x_rot = torch.stack((-x_odd, x_even), dim=-1) # [..., D//2, 2]
    return x_rot.flatten(-2)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    _, _, T, _ = x.shape

    # cos/sin: [T, D] -> [1, 1, T, D]
    cos = cos[:T][None, None, :, :]
    sin = sin[:T][None, None, :, :]

    return x * cos + rotate_half(x) * sin
