# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified from VGGT (https://github.com/facebookresearch/vggt) and SegVGGT
# (https://github.com/IDEA-Research/SegVGGT).

"""
Dense semantic feature head for query-based segmentation.

The head turns the frozen StreamVGGT token features into a dense per-pixel
feature map `semantic_feature_maps` of shape (B, S, h, w, D).  The object
queries produced by the aggregator are projected into the same D-dimensional
space and dot-producted with the feature map, which yields per-query mask
logits exactly like SegVGGT does for instance segmentation:

    query_masks = einsum("bqd,bshwd->bqshw", query_proj(object_queries), semantic_feature_maps)

For AIC (which only provides semantic labels) the per-query masks and per-query
class logits are combined into per-pixel semantic logits (Mask2Former-style).

Inspired by https://github.com/DepthAnything/Depth-Anything-V2
"""

from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import create_uv_grid, position_grid_to_embed


class SemanticHead(nn.Module):
    """
    Dense feature head for query-based segmentation.

    Args:
        dim_in (int): Input dimension (channels of the aggregated tokens, 2 * embed_dim).
        patch_size (int, optional): Patch size. Default is 14.
        num_classes (int, optional): Number of semantic classes (without the
            "no-object" slot). The query classifier outputs `num_classes + 1` logits.
        features (int, optional): Feature channels for intermediate representations. Default is 256.
        out_channels (List[int], optional): Output channels for each intermediate layer.
        intermediate_layer_idx (List[int], optional): Indices of aggregated token layers used by the head.
        pos_embed (bool, optional): Whether to use positional embedding. Default is True.
        feature_only (bool, optional): If True, return features only without the last several
            layers and activation head. Default is False.
        out_ratio (int, optional): Downscaling factor of the dense feature map w.r.t. the
            input image (4 means H/4 x W/4). Default is 4.
        maps_ratio (int or None, optional): Extra downscaling of the returned feature maps.
            Must be >= out_ratio. Default (None) keeps the maps at out_ratio.
        query_feat_dim (int, optional): Dimension of the per-pixel feature the queries are
            dot-producted with. Must equal `features // 2`. Default is 128.
        query_cls_hidden_dim (int, optional): Hidden dim of the query classification MLP.
        use_no_object (bool, optional): Whether the query classifier has a "no-object" slot.
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        num_classes: int = 11,
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        intermediate_layer_idx: Sequence[int] = (4, 11, 17, 23),
        pos_embed: bool = True,
        feature_only: bool = False,
        out_ratio: int = 4,
        maps_ratio: int = None,
        query_feat_dim: int = 128,
        query_cls_hidden_dim: int = 64,
        use_no_object: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.pos_embed = pos_embed
        self.feature_only = feature_only
        self.out_ratio = out_ratio
        self.maps_ratio = maps_ratio if maps_ratio is not None else out_ratio
        assert self.maps_ratio >= self.out_ratio, "maps_ratio must be >= out_ratio"
        self.intermediate_layer_idx = list(intermediate_layer_idx)
        self.num_classes = num_classes
        self.use_no_object = use_no_object
        self.num_query_classes = num_classes + (1 if use_no_object else 0)
        self.query_feat_dim = query_feat_dim
        assert features // 2 == query_feat_dim, (
            f"query_feat_dim must equal features // 2, got {query_feat_dim} vs {features // 2}"
        )

        self.norm = nn.LayerNorm(dim_in)

        # Projection layers for each output channel from tokens.
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0)
                for oc in out_channels
            ]
        )

        # Resize layers for upsampling feature maps.
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )

        self.scratch = _make_scratch(out_channels, features, expand=False)

        # Attach additional modules to scratch.
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        head_features_1 = features
        if feature_only:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1)
        else:
            self.scratch.output_conv1 = nn.Conv2d(
                head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
            )

        # Query classification MLP (operates on the query feature space).
        self.scratch.output_instance = nn.Sequential(
            nn.Linear(query_feat_dim, query_cls_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(query_cls_hidden_dim, self.num_query_classes),
        )

    # ------------------------------------------------------------------ #
    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 8,
    ) -> torch.Tensor:
        """
        Args:
            aggregated_tokens_list (List[Tensor]): list of token tensors, each (B, S, P, 2C)
            images (Tensor): input images (B, S, 3, H, W) in [0, 1]
            patch_start_idx (int): index where patch tokens start in the token sequence
            frames_chunk_size (int, optional): number of frames processed per chunk

        Returns:
            semantic_feature_maps (Tensor): (B, S, h, w, D) with h = H / maps_ratio
        """
        B, S, _, H, W = images.shape

        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)

        all_maps = []
        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)
            all_maps.append(
                self._forward_impl(aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx)
            )
        return torch.cat(all_maps, dim=1)

    # ------------------------------------------------------------------ #
    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ) -> torch.Tensor:
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, _, H, W = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out = []
        for dpt_idx, layer_idx in enumerate(self.intermediate_layer_idx):
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]

            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            x = x.reshape(B * S, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)
            out.append(x)

        out = self.scratch_forward(out)

        target_h = max(int(round(H / self.maps_ratio)), 1)
        target_w = max(int(round(W / self.maps_ratio)), 1)
        out = custom_interpolate(out, (target_h, target_w), mode="bilinear", align_corners=True)

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        if self.feature_only:
            return out.view(B, S, *out.shape[1:])

        # (B*S, D, h, w) -> (B, S, h, w, D)
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(B, S, *out.shape[1:])
        return out

    # ------------------------------------------------------------------ #
    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """Apply positional embedding to tensor x."""
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed.to(x.dtype)

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass through the fusion blocks."""
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4

        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1

        out = self.scratch.output_conv1(out)
        return out

    # ------------------------------------------------------------------ #
    def query_classifier(self, query_features: torch.Tensor) -> torch.Tensor:
        """Classify the object queries: (B, Q, D) -> (B, Q, num_query_classes)."""
        return self.scratch.output_instance(query_features)


################################################################################
# Modules
################################################################################


def _make_fusion_block(features: int, size: int = None, has_residual: bool = True, groups: int = 1) -> nn.Module:
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )


def _make_scratch(in_shape: Sequence[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape

    if expand:
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn, groups=1):
        super().__init__()
        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.norm1 = None
        self.norm2 = None
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block (DPT-style)."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
        has_residual=True,
        groups=1,
    ):
        super().__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups
        )

        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        output = xs[0]

        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)
        return output


def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """Custom interpolate to avoid INT_MAX issues in nn.functional.interpolate."""
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736
    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)
