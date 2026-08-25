from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from rl.networks.transformer.transformer import FeedForward
from rl.utils import resolve_nn_activation

from .utils import SinusoidalPositionalEmbedding

class TransformerXL(nn.Module):
    def __init__(
        self,
        segment_length: int,
        num_memory_segments: int,
        hidden_dim: int,
        num_heads: int,
        ff_dim: Optional[int] = None,
        num_layers: int = 4,
        activation: str = 'silu',
        dropout: float = 0.0,
        causal_mask: bool = False,
        block_variant: str = "postnorm",
    ):
        super().__init__()
        self.segment_length = segment_length
        self.num_memory_segments = num_memory_segments
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.transformer_xl = TransformerXLCore(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            activation=activation,
            dropout=dropout,
            causal_mask=causal_mask,
            block_variant=block_variant,
        )

        
    def forward(self, src: torch.Tensor):
        if src.ndim == 3:
            batch_size, seq_length, _ = src.size()
            segment_length = self.segment_length
            num_segments = seq_length // segment_length
            assert seq_length % self.segment_length == 0, "Sequence length must be divisible by segment length."
            src = src.view(batch_size, num_segments, segment_length, -1)
        elif src.ndim == 4:
            batch_size, num_segments, segment_length, _ = src.size()
        else:
            raise ValueError("Input src must be a 3D or 4D tensor.")
        
        segment_memory_cache = SegmentMemoryCache(
            num_layers=self.num_layers,
            segment_length=segment_length,
            num_memory_segments=self.num_memory_segments,
            hidden_dim=self.hidden_dim,
            batch_size=batch_size,
            device=src.device,
            dtype=src.dtype,
        )

        for i in range(num_segments):
            segment_memory = segment_memory_cache.get_memory()
            segment = src[:, i, :, :]
            transformer_output, segment_memory = self.transformer_xl(segment, mem=segment_memory) 
            segment_memory_cache.update(segment_memory)
        
        return transformer_output


class SegmentMemoryCache:
    def __init__(
        self,
        num_layers: int,
        segment_length: int,
        num_memory_segments: int,
        hidden_dim: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.segment_length = segment_length
        self.num_memory_segments = num_memory_segments
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.cache = torch.zeros(
            (num_memory_segments, num_layers, batch_size, segment_length, hidden_dim),
            device=device,
            dtype=dtype,
        )
        self.current_index = 0

    def update(self, segment_memory: torch.Tensor) -> None:
        if self.num_memory_segments == 0:
            return

        self.cache[:-1] = self.cache[1:]
        self.cache[-1] = segment_memory
        self.current_index = min(self.current_index + 1, self.num_memory_segments)
            
    def get_memory(self) -> Optional[torch.Tensor]:
        if self.current_index == 0 or self.num_memory_segments == 0:
            return None
        memory = self.cache[self.num_memory_segments - self.current_index :]
        memory = torch.cat(memory.unbind(dim=0), dim=2)
        return memory

class TransformerXLCore(nn.Module):
    def __init__(
            self, 
            num_layers,
            hidden_dim,
            num_heads,
            ff_dim=None,
            activation='silu',
            dropout=0.0,
            causal_mask=False,
            block_variant="postnorm",
        ):
        super().__init__()
        # TODO: add an embedding layer for input tokens
        self.layers = nn.ModuleList([
            TransformerXLLayer(
                hidden_dim,
                num_heads,
                ff_dim,
                activation,
                dropout,
                causal_mask=causal_mask,
                block_variant=block_variant,
            )
            for _ in range(num_layers)
        ])

    def forward(self, src, mem: Optional[torch.Tensor]=None):
        memory_cache = []
        for i, layer in enumerate(self.layers):
            # Cache layer input to align with Transformer-XL memory semantics.
            memory_cache.append(src.detach())
            src = layer(src, mem=mem[i] if mem is not None else None)
        memory_cache = torch.stack(memory_cache, dim=0)  # [num_layers, B, L, D]
        return src, memory_cache



class TransformerXLLayer(nn.Module):
    def __init__(
            self, 
            hidden_dim, 
            num_heads, 
            ff_dim=None, 
            activation='silu', 
            dropout=0.0, 
            causal_mask=False, 
            block_variant="postnorm",
        ):

        super().__init__()
        self.attn = RelativeMultiHeadAttention(hidden_dim, num_heads, dropout, causal_mask=causal_mask)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)


        self.ff = FeedForward(hidden_dim, ff_dim, activation, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.block_variant = block_variant
        if self.block_variant == "imp":
            self.activation1 = resolve_nn_activation(activation)
            self.activation2 = resolve_nn_activation(activation)

        elif self.block_variant == "gtrxl":
            self.activation1 = resolve_nn_activation(activation)
            self.activation2 = resolve_nn_activation(activation)
            self.gate1 = GRUGate(hidden_dim)
            self.gate2 = GRUGate(hidden_dim)
        elif self.block_variant == "postnorm":
            pass
        else:
            raise ValueError(f"Unsupported block variant: {block_variant}")

    def forward(self, src, mem: Optional[torch.Tensor]=None):


        if self.block_variant == "imp":
            q = self.norm1(src)
            if mem is None:
                k = q
            else:
                mem = self.norm1(mem.detach())
                k = torch.cat([mem, q], dim=1)

            attn_output = self.attn(q, k, k)
            attn_output = self.activation1(attn_output)
            attn_output = self.dropout1(attn_output)
            src = src + attn_output

            src_norm = self.norm2(src)
            ff_output = self.ff(src_norm)
            ff_output = self.dropout2(self.activation2(ff_output))
            src = src + ff_output
        elif self.block_variant == "gtrxl":
            q = self.norm1(src)
            if mem is None:
                k = q
            else:
                mem = self.norm1(mem.detach())
                k = torch.cat([mem, q], dim=1)

            attn_output = self.attn(q, k, k)
            attn_output = self.activation1(attn_output)
            attn_output = self.dropout1(attn_output)
            gated_attn_output = self.gate1(src, attn_output)
            src = gated_attn_output

            src_norm = self.norm2(src)
            ff_output = self.ff(src_norm)
            ff_output = self.activation2(ff_output)
            ff_output = self.dropout2(ff_output)
            gated_ff_output = self.gate2(src, ff_output)
            src = gated_ff_output

        else:
            q = src
            k = src if mem is None else torch.cat([mem.detach(), src], dim=1)

            attn_output = self.attn(q, k, k)
            attn_output = self.dropout1(attn_output)
            src = self.norm1(src + attn_output)

            ff_output = self.ff(src)
            ff_output = self.dropout2(ff_output)
            src = self.norm2(src + ff_output)

        return src

class OutputGating(nn.Module):
    def __init__(self, hidden_dim, b_g=1.0):
        super().__init__()
        self.W_g = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.constant_(self.W_g.bias, -b_g)

    def forward(self, x, y):
        gate = torch.sigmoid(self.W_g(x))
        return x + gate * y

class GRUGate(nn.Module):
    def __init__(self, input_dim: int, bg: float = 0.0):
        super(GRUGate, self).__init__()

        self.Wr = nn.Linear(input_dim, input_dim, bias=False)
        self.Ur = nn.Linear(input_dim, input_dim, bias=False)
        self.Wz = nn.Linear(input_dim, input_dim, bias=False)
        self.Uz = nn.Linear(input_dim, input_dim, bias=False)
        self.Wg = nn.Linear(input_dim, input_dim, bias=False)
        self.Ug = nn.Linear(input_dim, input_dim, bias=False)
        self.bg = nn.Parameter(torch.full([input_dim], bg))  # bias

        nn.init.xavier_uniform_(self.Wr.weight)
        nn.init.xavier_uniform_(self.Ur.weight)
        nn.init.xavier_uniform_(self.Wz.weight)
        nn.init.xavier_uniform_(self.Uz.weight)
        nn.init.xavier_uniform_(self.Wg.weight)
        nn.init.xavier_uniform_(self.Ug.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor):

        r = nn.Sigmoid()(self.Wr(y) + self.Ur(x))
        z = nn.Sigmoid()(self.Wz(y) + self.Uz(x) - self.bg)
        h = nn.Tanh()(self.Wg(y) + self.Ug(torch.mul(r, x)))
        return torch.mul(1 - z, x) + torch.mul(z, h)

class RelativeMultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        causal_mask=False,
    ):
        
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.causal_mask = causal_mask

        self.pos_emb = SinusoidalPositionalEmbedding(embed_dim)

        self.r_proj = nn.Linear(embed_dim, embed_dim)
        self.u_bias = nn.Parameter(torch.zeros(self.embed_dim))
        self.v_bias = nn.Parameter(torch.zeros(self.embed_dim))
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)


        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout_rate = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self._init_parameters()


    def _init_parameters(self) -> None:
        nn.init.normal_(self.u_bias, mean=0.0, std=0.02)
        nn.init.normal_(self.v_bias, mean=0.0, std=0.02)
    
    def _rel_shift(self, x: torch.Tensor, zero_triu: bool = False) -> torch.Tensor:
        """
        Transformer-XL rel_shift for logits.
        Args:
            x: [B, H, Lq, Lkq]
            L: current segment length
        Returns:
            shifted: [B, H, Lq, Lk]
        """
        Lq, Lkq = x.size()[-2:]
        Lk = Lkq - Lq
        Lm = Lk - Lq
        zero_pad = torch.zeros_like(x[..., :1])  # [B, H, Lq, 1]
        x_ = torch.cat([zero_pad, x], dim=-1)
        x_ = x_.view(*x.size()[:-2], Lkq + 1, Lq)          # [B, H, Lkq+1, Lq]
        x_ = x_[..., 1:, :]                   # [B, H, Lkq, Lq]
        x_ = x_.view(*x.size()[:-2], Lq, Lkq)              # [B, H, Lq, Lkq]

        x_ = x_[..., :Lk]                # [B, H, Lq, Lk]

        if zero_triu:
            ones = torch.ones_like(x_[..., :Lk])  # [B, H, Lq, Lk]
            x_ = x_ * torch.tril(ones, Lm)  # zero out upper triangular part

        return x_
    
    #region
    # def _rel_shift(self, x: torch.Tensor, zero_triu: bool = False) -> torch.Tensor:
    #     """
    #     Transformer-XL rel_shift for logits.
    #     Args:
    #         x: [B, H, Lq, Lk]
    #         L: current segment length
    #     Returns:
    #         shifted: [B, H, Lq, Lk]
    #     """
    #     Lq, Lk = x.size()[-2:]
    #     Lm = Lk - Lq
    #     zero_pad = torch.zeros_like(x[..., :1])  # [B, H, Lq, 1]

    #     x_padded = torch.cat([zero_pad, x], dim=-1)  # [B, H, Lq, Lk+1]
    #     x_padded = x_padded.view(*x.size()[:-2], Lk + 1, Lq)  # [B, H, Lk+1, Lq]

    #     x_ = x_padded[..., 1:, :]  # [B, H, Lk, Lq]
    #     x_ = x_.view(*x.size()[:-2], Lq, Lk)  # [B, H, Lq, Lk]

    #     if zero_triu:
    #         ones = torch.ones_like(x_[..., :Lk])  # [B, H, Lq, Lk]
    #         x_ = x_ * torch.tril(ones, Lm)  # zero out upper triangular part
        
    #     return x_

    # def forward1(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
    #     """
    #     Multi-head attention with optional Transformer-XL style relative bias.
    #     Args:
    #         q: [B, Lq, C]   current segment tokens
    #         k: [B, K, C]   memory + current segment (K = M + Lq)
    #         v: [B, K, C]   same layout as k
    #         attn_mask: [B, Lq, K]   attention mask
    #     Notes:

    #     """
    #     B, Lq, C = q.size()
    #     B, Lk, C = k.size()
    #     Lm = Lk - Lq
    #     assert Lk >= Lq
    #     assert k.size() == v.size()

    #     q = self.q_proj(q).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Lq, Dh]
    #     k = self.k_proj(k).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Lk, Dh]
    #     v = self.v_proj(v).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Lk, Dh]

    #     # Note: range includes Lk-1 down to 0
    #     pos_seq = torch.arange(Lk-1, -1, step=-1, dtype=torch.float32, device=q.device)
    #     rel_pos_emb = self.pos_emb(pos_seq).to(q.dtype)  # [L, C]
    
    #     r = self.r_proj(rel_pos_emb).view(Lk, self.num_heads, self.head_dim).permute(1, 0, 2)  # [H, Lk, Dh]

    #     u_bias = self.u_bias.view(self.num_heads, self.head_dim)  # [H, Dh]
    #     attn_ac = torch.einsum("bhqd, bhkd -> bhqk", q + u_bias[None, :, None, :], k)  # [B, H, Lq, Lk]

    #     v_bias = self.v_bias.view(self.num_heads, self.head_dim)  # [H, Dh]

    #     attn_bd = torch.einsum("bhqd, hkd -> bhqk", q+v_bias[None, :, None, :], r)  # [B, H, Lq, Lk]

    #     attn_bd = self._rel_shift(attn_bd)

    #     attn_scores = (attn_ac + attn_bd)  / (self.head_dim ** 0.5)  # [B, H, Lq, Lk]

    #     if self.causal_mask:
    #         causal_mask = torch.triu(torch.ones((Lq, Lk), device=q.device), diagonal=1+Lm).bool()  # [Lq, Lk]
    #         attn_scores = attn_scores.masked_fill(causal_mask[None, None, :, :], float("-inf"))

    #     attn_probs = torch.softmax(attn_scores, dim=-1)
    #     attn_probs = self.attn_dropout(attn_probs)

    #     attn_output = torch.einsum("bhqk, bhkd -> bhqd", attn_probs, v)  # [B, num_heads, Lq, head_dim]

    #     attn_output = attn_output.transpose(1, 2).contiguous().view(B, Lq, C)  # [B, Lq, embed_dim]

    #     attn_output = self.out_proj(attn_output)
    #     attn_output = self.proj_dropout(attn_output)

    #     return attn_output
    
    # def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
    #     attn_output1 = self.forward1(q, k, v, attn_mask)
    #     attn_output2 = self.forward2(q, k, v, attn_mask)
    #     assert torch.allclose(attn_output1, attn_output2), "forward1 and forward2 outputs do not match!"
    #     return attn_output1
    # endregion
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Multi-head attention with optional Transformer-XL style relative bias.
        Args:
            q: [B, Lq, C]   current segment tokens
            k: [B, K, C]   memory + current segment (K = M + Lq)
            v: [B, K, C]   same layout as k
            attn_mask: [B, Lq, K]   attention mask
        Notes:

        """
        B, Lq, C = q.size()
        B, Lk, C = k.size()
        Lm = Lk - Lq
        Lo = Lk + Lq
        assert Lk >= Lq
        assert k.size() == v.size()

        q = self.q_proj(q).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Lq, Dh]
        k = self.k_proj(k).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Lk, Dh]
        v = self.v_proj(v).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Lk, Dh]

        # Note: range includes Lk-1 down to -Lq
        pos_seq = torch.arange(Lk-1, -Lq-1, step=-1, dtype=torch.float32, device=q.device)
        rel_pos_emb = self.pos_emb(pos_seq).to(q.dtype)  # [Lo, C]

        r = self.r_proj(rel_pos_emb).view(Lo, self.num_heads, self.head_dim).permute(1, 0, 2)  # [H, Lo, Dh]

        u_bias = self.u_bias.view(self.num_heads, self.head_dim)  # [H, Dh]
        attn_ac = torch.einsum("bhqd, bhkd -> bhqk", q + u_bias[None, :, None, :], k)  # [B, H, Lq, Lk]

        v_bias = self.v_bias.view(self.num_heads, self.head_dim)  # [H, Dh]

        attn_bd = torch.einsum("bhqd, hod -> bhqo", q+v_bias[None, :, None, :], r)  # [B, H, Lq, Lo]

        attn_bd = self._rel_shift(attn_bd)

        attn_scores = (attn_ac + attn_bd)  / (self.head_dim ** 0.5)  # [B, H, Lq, Lk]

        if self.causal_mask:
            causal_mask = torch.triu(torch.ones((Lq, Lk), device=q.device), diagonal=1+Lm).bool()  # [Lq, Lk]
            attn_scores = attn_scores.masked_fill(causal_mask[None, None, :, :], float("-inf"))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        attn_output = torch.einsum("bhqk, bhkd -> bhqd", attn_probs, v)  # [B, num_heads, Lq, head_dim]

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Lq, C)  # [B, Lq, embed_dim]

        attn_output = self.out_proj(attn_output)
        attn_output = self.proj_dropout(attn_output)

        return attn_output



if __name__ == "__main__":
    # Quick visual check for _rel_shift behavior
    embed_dim = 8
    B, H, L = 1, 1, 3
    M = 4 
    K = M + L
    KL = K + L

    attn = RelativeMultiHeadAttention(embed_dim=embed_dim, num_heads=H)

    q = torch.zeros((B, L, embed_dim))
    k = torch.zeros((B, K, embed_dim))
    v = torch.zeros((B, K, embed_dim))
    attn_output = attn(q, k, v)
    
