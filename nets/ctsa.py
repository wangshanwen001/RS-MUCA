import numpy as np
import torch
import torch.nn as nn
from torch.nn import Dropout, Softmax, LayerNorm
from einops import rearrange, repeat
import torch.nn.functional as F
import math

class CrossAttnMem(nn.Module):
    def __init__(self, num_heads, embedding_channels, attention_dropout_rate, num_class, patch_num):
        super().__init__()
        self.KV_size = embedding_channels * num_heads
        self.num_class = num_class
        self.patch_num = patch_num
        self.num_heads = num_heads
        self.attention_head_size = embedding_channels 
        self.q_u = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.k_u = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.v_u = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)

        self.q_l2u = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.k_l2u = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.v_l2u = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)

        self.psi = nn.InstanceNorm2d(self.num_heads)
        self.softmax = Softmax(dim=3)

        self.out_u = nn.Linear(embedding_channels * self.num_heads, embedding_channels, bias=False)
        self.out_l2u = nn.Linear(embedding_channels * self.num_heads, embedding_channels, bias=False)
        self.attn_dropout = Dropout(attention_dropout_rate)
        self.proj_dropout = Dropout(attention_dropout_rate)
        self.pseudo_label = None
        self.SMem_size = embedding_channels
        self.register_buffer("queue_ptr", torch.zeros(self.num_class, dtype=torch.long))
        self.register_buffer("kv_queue", torch.randn(self.num_class, self.SMem_size, self.patch_num))

    def multi_head_rep(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, emb):

        emb_s, emb_t = torch.split(emb, emb.size(0) // 2, dim=0)
        _, N, C = emb_t.size()

        q_l2u = self.q_l2u(emb_s)
        k_l2u = self.k_l2u(emb_t)
        v_l2u = self.v_l2u(emb_t)

        batch_size = q_l2u.size(0)

        k_l2u = rearrange(k_l2u, 'b n c -> n (b c)')
        v_l2u = rearrange(v_l2u, 'b n c -> n (b c)')

        k_l2u = repeat(k_l2u, 'n bc -> r n bc', r=batch_size)
        v_l2u = repeat(v_l2u, 'n bc -> r n bc', r=batch_size)

        q_l2u = q_l2u.unsqueeze(1).transpose(-1, -2)
        k_l2u = k_l2u.unsqueeze(1)
        v_l2u = v_l2u.unsqueeze(1).transpose(-1, -2)


        cross_attn_l2u = torch.matmul(q_l2u, k_l2u)
        cross_attn_l2u = self.attn_dropout(self.softmax(self.psi(cross_attn_l2u)))
        cross_attn_l2u = torch.matmul(cross_attn_l2u, v_l2u)

        cross_attn_l2u = cross_attn_l2u.permute(0, 3, 2, 1).contiguous()
        new_shape_l2u = cross_attn_l2u.size()[:-2] + (self.KV_size,)
        cross_attn_l2u = cross_attn_l2u.view(*new_shape_l2u)

        out_l2u = self.out_l2u(cross_attn_l2u)
        out_l2u = self.proj_dropout(out_l2u)

        return out_l2u

class CTSA(nn.Module):
    def __init__(self, num_heads, embedding_channels, channel_num, channel_num_out,
                 attention_dropout_rate, num_class, patch_num):
        super().__init__()
        self.map_in = nn.Sequential(nn.Conv2d(channel_num, embedding_channels, kernel_size=1, padding=0),
                                     nn.GELU())
        self.attn_norm = LayerNorm(embedding_channels, eps=1e-6)
        self.attn = CrossAttnMem(num_heads, embedding_channels, attention_dropout_rate, num_class, patch_num)
        self.encoder_norm = LayerNorm(embedding_channels, eps=1e-6)
        self.map_out = nn.Sequential(nn.Conv2d(embedding_channels, channel_num_out, kernel_size=1, padding=0),
                                     nn.GELU())
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, en):
        if not self.training:
            en = torch.cat((en, en))

        _, _, h, w = en.shape
        en = self.map_in(en)
        en = en.flatten(2).transpose(-1, -2)  # (B, n_patches, hidden)  #

        emb = self.attn_norm(en)
        emb = self.attn(emb)

        en_s, en_t = torch.split(en, en.size(0) // 2, dim=0)
        emb = emb + en_s

        out = self.encoder_norm(emb)

        B, n_patch, hidden = out.size()
        out = out.permute(0, 2, 1).contiguous().view(B, hidden, h, w)

        out = self.map_out(out)

        if not self.training:
            out = torch.split(out, out.size(0) // 2, dim=0)[0]

        return out

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

