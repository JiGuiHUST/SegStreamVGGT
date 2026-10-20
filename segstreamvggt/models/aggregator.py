# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict, Any

from segstreamvggt.layers import PatchEmbed
from segstreamvggt.layers.block import Block, CrossBlock
from segstreamvggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from segstreamvggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer, with
    StreamVGGT's causal (streaming) global attention and an optional
    SegVGGT-style object-query branch for query-based segmentation.


    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
        enable_query_seg (bool): Whether to enable the object-query segmentation branch.
        query_num (int): Number of learnable object queries.
        query_dim (int or None): Dimension of the object queries. If None, uses embed_dim.
        cross_block_layers (list[int] or None): Global block indices at which the object queries
            cross-attend to the backbone tokens. If None, all global blocks are used.
        cross_block_params (dict or None): Custom parameters for the query cross-attention blocks.
        query_self_attention (bool): Whether to use self-attention among object queries.
        query_sa_layers (list[int] or None): Layers at which query self-attention is applied.
        query_sa_params (dict or None): Custom parameters for the query self-attention blocks.
        supervise_attend_frame (bool): Whether to expose per-frame attention mass (FADA-style).
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        enable_query_seg=False,
        query_num=100,
        query_dim=None,
        cross_block_layers=None,
        cross_block_params=None,
        query_self_attention=False,
        query_sa_layers=None,
        query_sa_params=None,
        supervise_attend_frame=False,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # ---- SegVGGT-style object-query segmentation branch -------------------
        self.enable_query_seg = enable_query_seg
        if enable_query_seg:
            self.query_num = query_num
            self.query_dim = query_dim if query_dim is not None else embed_dim
            self.query_token = nn.Parameter(torch.randn(1, query_num, self.query_dim))
            nn.init.normal_(self.query_token, std=1e-6)

            if cross_block_layers is not None:
                self.cross_block_layers = list(cross_block_layers)
            else:
                self.cross_block_layers = list(range(depth))
            self.cross_block_params = cross_block_params or {}
            self.query_cross_blocks = nn.ModuleList(
                [
                    CrossBlock(
                        dim=self.cross_block_params.get("embed_dim", embed_dim),
                        num_heads=self.cross_block_params.get("num_heads", num_heads),
                        dim_q=self.query_dim,
                        dim_kv=embed_dim,
                        mlp_ratio=self.cross_block_params.get("mlp_ratio", mlp_ratio),
                        qkv_bias=self.cross_block_params.get("qkv_bias", qkv_bias),
                        proj_bias=self.cross_block_params.get("proj_bias", proj_bias),
                        ffn_bias=self.cross_block_params.get("ffn_bias", ffn_bias),
                        init_values=self.cross_block_params.get("init_values", init_values),
                        qk_norm=self.cross_block_params.get("qk_norm", qk_norm),
                    )
                    if i in self.cross_block_layers
                    else None
                    for i in range(depth)
                ]
            )

            self.query_self_attention = query_self_attention
            self.query_sa_params = query_sa_params or {}
            self.query_sa_layers = list(query_sa_layers) if query_sa_layers is not None else list(range(depth))
            if query_self_attention:
                self.query_self_attn = nn.ModuleList(
                    [
                        block_fn(
                            dim=self.query_dim,
                            num_heads=self.query_sa_params.get("num_heads", num_heads),
                            mlp_ratio=self.query_sa_params.get("mlp_ratio", mlp_ratio),
                            qkv_bias=self.query_sa_params.get("qkv_bias", qkv_bias),
                            proj_bias=self.query_sa_params.get("proj_bias", proj_bias),
                            ffn_bias=self.query_sa_params.get("ffn_bias", ffn_bias),
                            init_values=self.query_sa_params.get("init_values", init_values),
                            qk_norm=self.query_sa_params.get("qk_norm", qk_norm),
                            rope=None,
                        )
                        if i in self.query_sa_layers
                        else None
                        for i in range(depth)
                    ]
                )
            else:
                self.query_self_attn = None

            self.supervise_attend_frame = supervise_attend_frame

            # projection of the object queries into the dense-feature space used
            # for the dot-product mask head (SegVGGT uses a fixed 128-d space)
            self.query_feat_dim = self.cross_block_params.get("query_feat_dim", 128)
            self.query_proj = nn.Linear(self.query_dim, self.query_feat_dim)

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).reshape(1, 1, 3, 1, 1),
                persistent=False,
            )


    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(
        self,
        images: torch.Tensor,
        past_key_values=None,
        use_cache=False,
        past_frame_idx=0
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if use_cache and past_key_values[0] is not None:
            _, _, S_true, _, _ = past_key_values[0][0].shape
            S_true += 1
        else:
            S_true = S
        
        if use_cache and S > 1:
            print(f"Use KV cache expects S=1, got S={S}")

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean.to(images.device)) / self._resnet_std.to(images.device)

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.reshape(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        if use_cache:
            camera_token_full = slice_expand_and_flatten(self.camera_token, B, S_true)
            camera_token = camera_token_full[-1:, :, :]
            
            register_token_full = slice_expand_and_flatten(self.register_token, B, S_true)
            register_token = register_token_full[-1:, :, :]
        else:
            camera_token = slice_expand_and_flatten(self.camera_token, B, S)
            register_token = slice_expand_and_flatten(self.register_token, B, S)
        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        # ---- object queries (kept OUTSIDE the token sequence, like SegVGGT) ----
        if self.enable_query_seg:
            object_queries = self.query_token.expand(B, -1, -1).contiguous()  # (B, Q, C_q)
        query_attn_frame_mean_list = []

        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    if use_cache:
                        if past_key_values[global_idx] is not None:
                            k, v = past_key_values[global_idx]
                        tokens, global_idx, global_intermediates, new_kv = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos,
                            past_key_values_block=past_key_values[global_idx] if past_key_values[global_idx] is not None else None,
                            use_cache=True,
                            past_frame_idx=past_frame_idx
                        )
                        past_key_values[global_idx - 1] = new_kv
                    else: 
                        tokens, global_idx, global_intermediates = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos
                        )

                    # ---- update object queries after each global block ----
                    if self.enable_query_seg:
                        cross_idx = global_idx - 1
                        if cross_idx in self.cross_block_layers:
                            object_queries, attn_frame_mean = self._process_query_cross_attention(
                                tokens, object_queries, B, S, P, C, cross_idx
                            )
                            if self.supervise_attend_frame and attn_frame_mean is not None:
                                query_attn_frame_mean_list.append(attn_frame_mean)
                        if self.query_self_attention and cross_idx in self.query_sa_layers:
                            object_queries = self._process_query_self_attention(object_queries, cross_idx)
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")
            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates

        if self.enable_query_seg:
            if use_cache:
                return output_list, self.patch_start_idx, past_key_values, object_queries, query_attn_frame_mean_list
            return output_list, self.patch_start_idx, object_queries, query_attn_frame_mean_list
        if use_cache:      
            return output_list, self.patch_start_idx, past_key_values
        return output_list, self.patch_start_idx

    def _process_query_cross_attention(self, tokens, object_queries, B, S, P, C, cross_idx):
        """Update the object queries by cross-attending to the backbone tokens.

        tokens: (B, S*P, C) or (B*S, P, C); object_queries: (B, Q, C_q)
        Returns: (updated queries, per-frame attention mass or None)
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.reshape(B, S, P, C).reshape(B, S * P, C)

        object_queries, attn_frame_mean = self.query_cross_blocks[cross_idx](
            object_queries, tokens, P, self.supervise_attend_frame, None, cross_idx
        )
        return object_queries, attn_frame_mean

    def _process_query_self_attention(self, object_queries, cross_idx):
        """Self-attention among the object queries."""
        object_queries = self.query_self_attn[cross_idx](object_queries, pos=None)
        return object_queries

    def query_named_parameters(self):
        """Yield the names of the parameters that belong to the query branch."""
        for name, _ in self.named_parameters():
            if name.startswith("query_token") or name.startswith("query_cross_blocks") \
                    or name.startswith("query_self_attn") or name.startswith("query_proj"):
                yield name

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.reshape(B, S, P, C).reshape(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.reshape(B, S, P, 2).reshape(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.reshape(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(
        self,
        tokens,
        B,
        S,
        P,
        C,
        global_idx,
        pos=None,
        past_key_values_block=None,
        use_cache=False,
        past_frame_idx=0
    ) -> Union[Tuple[torch.Tensor, int, List[torch.Tensor]], Tuple[torch.Tensor, int, List[torch.Tensor], List]]:
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
                """
        
        if tokens.shape != (B, S * P, C):
            tokens = tokens.reshape(B, S, P, C).reshape(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.reshape(B, S, P, 2).reshape(B, S * P, 2)
            
        intermediates = []

        for _ in range(self.aa_block_size):
            if not use_cache:
                L = S * P
                frame_ids = torch.arange(L, device=tokens.device) // P  # [0,0,...,1,1,...,S-1]
                future_frame = frame_ids.unsqueeze(1) < frame_ids.unsqueeze(0)
                attn_mask = future_frame.to(tokens.dtype) * torch.finfo(tokens.dtype).min
            else:
                attn_mask = None
                
            if use_cache:
                tokens, block_kv = self.global_blocks[global_idx](
                    tokens, 
                    pos=pos, 
                    attn_mask=attn_mask, 
                    past_key_values=past_key_values_block,
                    use_cache=True
                )
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos, attn_mask=attn_mask)
            global_idx += 1
            intermediates.append(tokens.reshape(B, S, P, C))

            # if self.use_causal_global:
            #     del attn_mask
        if use_cache:
            return tokens, global_idx, intermediates, block_kv
        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.reshape(B * S, *combined.shape[2:])
    return combined
