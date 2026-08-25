import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim ):
        super(SinusoidalPositionalEmbedding, self).__init__()

        self.embed_dim = embed_dim

        inv_freq = 1 / (10000 ** (torch.arange(0.0, embed_dim, 2.0) / embed_dim))
        self.register_buffer('inv_freq', inv_freq)  # [embed_dim/2]

    def forward(self, pos_seq):
        # pos_seq: [L]
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)  # [L, embed_dim/2]
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)  # [L, embed_dim]

        return pos_emb