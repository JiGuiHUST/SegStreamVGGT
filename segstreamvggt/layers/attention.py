import logging
import os
import warnings

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Union, Tuple, Dict, Optional

from einops import rearrange

XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, 
        x: torch.Tensor, 
        pos=None, 
        attn_mask=None, 
        past_key_values=None, 
        use_cache=False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple]]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        pos_k = pos
        if use_cache:
            k = k.unsqueeze(2)
            v = v.unsqueeze(2)
            if past_key_values is not None:
                past_k, past_v = past_key_values
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
                
            new_kv = (k, v)
            a, b, c, d, e = k.shape
            k = k.reshape(a, b, c*d, e)
            v = v.reshape(a, b, c*d, e)
            if pos_k is not None:
                #print(pos_k.shape)
                pos_k = pos_k.repeat(1, c, 1)
                #print(pos_k.shape)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos_k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )

        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

            # Mask
            if attn_mask is not None:
                assert attn_mask.shape[-2:] == (N, N), f"Expected mask shape [..., {N}, {N}], got {attn_mask.shape}"
                attn = attn + attn_mask

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if use_cache:
            return x, new_kv
        return x


class CrossAttention(nn.Module):
    """
    Cross-attention module where query comes from one tensor and key/value come
    from another tensor.

    Adapted from SegVGGT (https://github.com/IDEA-Research/SegVGGT) so that the
    object-query branch can be grafted onto StreamVGGT.

    Args:
        dim: Output dimension (and default input dimension for q/k/v if not specified separately)
        dim_q: Input dimension for query. If None, uses `dim`.
        dim_kv: Input dimension for key/value. If None, uses `dim`.
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias in q/k/v projection layers
        proj_bias: Whether to use bias in output projection layer
        attn_drop: Dropout rate for attention weights
        proj_drop: Dropout rate for output projection
        norm_layer: Normalization layer class
        qk_norm: Whether to apply normalization to q and k
        fused_attn: Whether to use fused attention implementation
    """

    def __init__(
        self,
        dim: int,
        dim_q: int = None,
        dim_kv: int = None,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,
    ) -> None:
        super().__init__()
        self.dim_q = dim_q if dim_q is not None else dim
        self.dim_kv = dim_kv if dim_kv is not None else dim
        self.dim = dim  # attention/working dimension

        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.q_proj = nn.Linear(self.dim_q, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(self.dim_kv, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(self.dim_kv, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, self.dim_q, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        P: int,
        supervise_attend_frame: bool = False,
        attn_mask: Tensor = None,
        layer_id: int = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Args:
            x_q: Query tensor of shape (B, N_q, dim_q)
            x_kv: Key/Value tensor of shape (B, N_kv, dim_kv)
            P: int - number of tokens per frame
            supervise_attend_frame: bool - whether to return frame-level attention mass
            attn_mask: Attention mask of shape (B, N_q, N_kv), True means masked (not attended to)

        Returns:
            Tuple(Tensor, Tensor):
                - Output tensor of shape (B, N_q, dim_q)
                - Frame-level attention mass (B, N_q, num_frames), or None
        """
        B, N_q, C_q = x_q.shape
        B_kv, N_kv, C_kv = x_kv.shape
        assert B == B_kv, "Batch size must match"
        assert C_q == self.dim_q, f"Query dimension mismatch: expected {self.dim_q}, got {C_q}"
        assert C_kv == self.dim_kv, f"Key/Value dimension mismatch: expected {self.dim_kv}, got {C_kv}"
        num_frames = max(N_kv // P, 1)
        # When KV length is not a multiple of P (e.g. cached/streaming mode), treat
        # the whole KV sequence as a single "frame" for the attention-mass readout.
        frame_level_ok = (N_kv % P == 0) and (P > 0)

        q = self.q_proj(x_q).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x_kv).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x_kv).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q, k = self.q_norm(q), self.k_norm(k)

        if attn_mask is not None:
            attn_bias = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn_bias = torch.zeros_like(attn_bias, dtype=q.dtype).masked_fill(attn_bias, float("-inf"))
        else:
            attn_bias = None

        if self.fused_attn and (not supervise_attend_frame):
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
            attn_frame_mean = None
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attn_bias is not None:
                attn = attn + attn_bias
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

            if supervise_attend_frame and frame_level_ok:
                attn_mean = attn.mean(dim=1)                          # (B, N_q, N_kv)
                attn_mean = attn_mean.view(B, N_q, num_frames, P)
                attn_frame_mean = attn_mean.mean(dim=-1)              # (B, N_q, num_frames)
            else:
                attn_frame_mean = None

        x = x.transpose(1, 2).reshape(B, N_q, self.dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn_frame_mean


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)

        return x
