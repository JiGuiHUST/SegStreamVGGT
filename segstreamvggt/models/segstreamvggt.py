# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# SegStreamVGGT: query-based semantic segmentation for multiple aerial datasets,
# built on StreamVGGT (https://github.com/wzzheng/StreamVGGT) and imitating the
# structure of SegVGGT (https://github.com/IDEA-Research/SegVGGT).

import torch
import torch.nn as nn
import torch.nn.functional as F

from segstreamvggt.models.aggregator import Aggregator
from segstreamvggt.heads.camera_head import CameraHead
from segstreamvggt.heads.dpt_head import DPTHead
from segstreamvggt.heads.track_head import TrackHead
from segstreamvggt.heads.semantic_head import SemanticHead


def combine_query_predictions(
    query_masks: torch.Tensor,
    query_logits: torch.Tensor,
    drop_no_object: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mask2Former-style combination of per-query masks and class logits.

    Args:
        query_masks: (B, Q, S, h, w) per-query mask logits
        query_logits: (B, Q, C+1) per-query class logits; the last slot is "no object"
        drop_no_object: drop the "no object" slot before combining
        eps: numerical floor for the normalisation

    Returns:
        seg_prob: (B, C, S, h, w) per-pixel class probabilities (sum to 1 over classes)
    """
    mask_prob = torch.sigmoid(query_masks.float())                     # (B, Q, S, h, w)
    cls_prob = F.softmax(query_logits.float(), dim=-1)                 # (B, Q, C + 1)
    if drop_no_object:
        cls_prob = cls_prob[..., :-1]
    seg_prob = torch.einsum("bqc,bqshw->bcshw", cls_prob, mask_prob)   # (B, C, S, h, w)
    seg_prob = seg_prob / seg_prob.sum(dim=1, keepdim=True).clamp_min(eps)
    return seg_prob


class SegStreamVGGT(nn.Module):
    """StreamVGGT backbone + SegVGGT-style object-query segmentation branch.

    The (optionally frozen) StreamVGGT aggregator produces geometry-aware token
    features; a set of learnable object queries iteratively cross-attends to
    those tokens and is finally decoded into per-query masks (dot product with a
    dense DPT feature map) and per-query class logits.  For semantic
    segmentation the two are combined into per-pixel class probabilities.

    Args:
        img_size (int): Image size in pixels (must be divisible by patch_size).
        patch_size (int): Patch size (14 for StreamVGGT/VGGT).
        embed_dim (int): Token embedding dimension (1024).
        num_classes (int): Number of semantic classes (AIC-11 -> 12, incl. ignore id 0).
        enable_camera/point/depth/track (bool): keep the original StreamVGGT heads.
        query_num (int): number of learnable object queries.
        query_dim (int | None): dimension of the object queries (defaults to embed_dim).
        cross_block_layers (list[int] | None): global blocks at which queries cross-attend.
        query_self_attention (bool): enable self-attention among queries.
        supervise_attend_frame (bool): return frame-level attention mass (FADA-style).
        out_ratio (int): resolution divisor of the dense feature map (H / out_ratio).
        maps_ratio (int | None): requested (B,S,h,w,D) feature map resolution divisor.
        ignore_index (int): label id ignored by the loss (0 = unlabeled).
    """

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        num_classes: int = 12,
        ignore_index: int = 0,
        enable_camera: bool = False,
        enable_point: bool = False,
        enable_depth: bool = False,
        enable_track: bool = False,
        query_num: int = 100,
        query_dim: int = 256,
        cross_block_layers=None,
        cross_block_params=None,
        query_self_attention: bool = False,
        query_sa_layers=None,
        query_sa_params=None,
        supervise_attend_frame: bool = False,
        out_ratio: int = 4,
        maps_ratio: int = None,
        query_cls_hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_ratio = out_ratio

        # A compact working dimension for the query cross-attention blocks:
        # SegVGGT instantiates one cross block per global block, which is very
        # heavy at the full 1024-d width. 256-d queries + 256-d attention
        # (Mask2Former-style) keeps the trainable branch small while preserving
        # the architecture (queries read from every global block).
        if cross_block_params is None:
            cross_block_params = {"embed_dim": 256, "num_heads": 8, "mlp_ratio": 2.0}

        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            enable_query_seg=True,
            query_num=query_num,
            query_dim=query_dim,
            cross_block_layers=cross_block_layers,
            cross_block_params=cross_block_params,
            query_self_attention=query_self_attention,
            query_sa_layers=query_sa_layers,
            query_sa_params=query_sa_params,
            supervise_attend_frame=supervise_attend_frame,
        )

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = (
            DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
            if enable_point
            else None
        )
        self.depth_head = (
            DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
            if enable_depth
            else None
        )
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

        self.semantic_head = SemanticHead(
            dim_in=2 * embed_dim,
            patch_size=patch_size,
            num_classes=num_classes,
            out_ratio=out_ratio,
            maps_ratio=maps_ratio,
            query_cls_hidden_dim=query_cls_hidden_dim,
        )

        self.num_queries = query_num

    # ------------------------------------------------------------------ #
    def forward(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor = None,
        past_key_values=None,
        use_cache: bool = False,
        past_frame_idx: int = 0,
        with_geometry: bool = None,
    ):
        """
        Args:
            images (torch.Tensor): (S, 3, H, W) or (B, S, 3, H, W) in [0, 1].
            query_points: optional point queries for the tracking head.
            with_geometry (bool): whether to also run the camera/point/depth/track
                heads (default True when any of them is enabled).

        Returns dict with:
            - seg_prob (B, C, S, h, w): per-pixel class probabilities (query-based)
            - seg_logits (B, S, C, h, w): log of seg_prob (for CE loss)
            - query_masks (B, Q, S, h, w): per-query mask logits
            - query_logits (B, Q, C+1): per-query class logits
            - semantic_feature_maps (B, S, h, w, D)
            - object_queries (B, Q, C_q)
            - attn_frame_mean (L, B, Q, S) if supervise_attend_frame
            - geometry outputs (depth/world_points/pose_enc) if enabled
        """
        if images.dim() == 4:
            images = images.unsqueeze(0)
        if query_points is not None and query_points.dim() == 2:
            query_points = query_points.unsqueeze(0)

        agg_out = self.aggregator(
            images,
            past_key_values=past_key_values,
            use_cache=use_cache,
            past_frame_idx=past_frame_idx,
        )
        if len(agg_out) == 5:  # cached + query seg
            aggregated_tokens_list, patch_start_idx, past_key_values, object_queries, attn_frame_mean_list = agg_out
        elif len(agg_out) == 4:  # query seg
            aggregated_tokens_list, patch_start_idx, object_queries, attn_frame_mean_list = agg_out
        elif len(agg_out) == 3:  # cached, no query seg
            aggregated_tokens_list, patch_start_idx, past_key_values = agg_out
            object_queries, attn_frame_mean_list = None, []
        else:
            aggregated_tokens_list, patch_start_idx = agg_out
            object_queries, attn_frame_mean_list = None, []

        predictions = {"aggregated_tokens_list": aggregated_tokens_list, "patch_start_idx": patch_start_idx}

        if with_geometry is None:
            with_geometry = any(
                head is not None
                for head in (self.camera_head, self.point_head, self.depth_head, self.track_head)
            )

        with torch.cuda.amp.autocast(enabled=False):
            if with_geometry:
                if self.camera_head is not None:
                    pose_enc_list = self.camera_head(aggregated_tokens_list)
                    predictions["pose_enc"] = pose_enc_list[-1]
                if self.depth_head is not None:
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                    )
                    predictions["depth"] = depth
                    predictions["depth_conf"] = depth_conf
                if self.point_head is not None:
                    pts3d, pts3d_conf = self.point_head(
                        aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                    )
                    predictions["world_points"] = pts3d
                    predictions["world_points_conf"] = pts3d_conf
                if self.track_head is not None and query_points is not None:
                    track_list, vis, conf = self.track_head(
                        aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                    )
                    predictions["track"] = track_list[-1]
                    predictions["vis"] = vis
                    predictions["conf"] = conf

            semantic_feature_maps = self.semantic_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            )  # (B, S, h, w, D)
            query_features = self.aggregator.query_proj(object_queries)            # (B, Q, D)
            query_masks = torch.einsum("bqd,bshwd->bqshw", query_features, semantic_feature_maps)
            query_logits = self.semantic_head.query_classifier(query_features)     # (B, Q, C+1)

            seg_prob = combine_query_predictions(query_masks, query_logits)
            predictions["semantic_feature_maps"] = semantic_feature_maps
            predictions["object_queries"] = object_queries
            predictions["query_features"] = query_features
            predictions["query_masks"] = query_masks
            predictions["query_logits"] = query_logits
            predictions["seg_prob"] = seg_prob
            predictions["seg_logits"] = torch.log(seg_prob.clamp_min(1e-8))

        if self.aggregator.supervise_attend_frame and len(attn_frame_mean_list) > 0:
            predictions["attn_frame_mean"] = torch.stack(attn_frame_mean_list, dim=0)

        if use_cache:
            predictions["past_key_values"] = past_key_values

        if not self.training:
            predictions["images"] = images

        return predictions

    # ------------------------------------------------------------------ #
    def seg_branch_named_parameters(self):
        """Names of the parameters that belong to the query segmentation branch."""
        names = set()
        for name, _ in self.semantic_head.named_parameters():
            names.add("semantic_head." + name)
        for name in self.aggregator.query_named_parameters():
            names.add("aggregator." + name)
        return names

    # ------------------------------------------------------------------ #
    # 训练范围 (哪些参数可训练) —— 默认只训"分割部分", 几何重建部分全部冻结
    # ------------------------------------------------------------------ #
    GEOMETRY_PREFIXES = ("depth_head.", "point_head.", "camera_head.", "track_head.")

    def train_scope_names(self, scope: str = "seg") -> set:
        """scope 对应的**可训练**参数名集合。

        * ``"seg"``          —— 只训分割分支: 语义头 + object query 分支（默认）
        * ``"seg+geometry"`` —— 额外训 depth/point head（几何重建头，需 --with-geometry 1 才会被构建）
        * ``"all"``          —— 全部可训练（含聚合器主干，等价于全量微调）
        """
        names = set(self.seg_branch_named_parameters())
        if scope == "all":
            return {n for n, _ in self.named_parameters()}
        if scope in ("seg+geometry", "geometry"):
            for n, _ in self.named_parameters():
                if n.startswith(("depth_head.", "point_head.")):
                    names.add(n)
        return names

    def set_train_scope(self, scope: str = "seg"):
        """按 scope 设置 requires_grad（其余全部冻结）。返回可训练参数名集合。"""
        keep = self.train_scope_names(scope)
        for name, param in self.named_parameters():
            param.requires_grad_(name in keep)
        self.train_scope = scope
        return keep

    def set_backbone_trainable(self, trainable: bool = False):
        """Freeze/unfreeze the StreamVGGT backbone (aggregator + geometry heads).

        等价于 set_train_scope("seg") / set_train_scope("all") 的简化形式（保留旧接口）。
        """
        if trainable:
            self.set_train_scope("all")
            return
        keep = self.seg_branch_named_parameters()
        for name, param in self.named_parameters():
            param.requires_grad_(name in keep)
        self.train_scope = "seg"

    def parameter_groups_report(self) -> dict:
        """按模块统计参数量与可训练情况 (用于日志/自检)。"""
        groups = {
            "aggregator(主干)": "aggregator.",
            "aggregator.query(分割)": "aggregator.query",
            "semantic_head(分割)": "semantic_head.",
            "depth_head(几何)": "depth_head.",
            "point_head(几何)": "point_head.",
            "camera_head(几何)": "camera_head.",
            "track_head(几何)": "track_head.",
        }
        out = {}
        for label, prefix in groups.items():
            tot = tr = 0
            for n, p in self.named_parameters():
                if not n.startswith(prefix):
                    continue
                if label == "aggregator(主干)" and n.startswith("aggregator.query"):
                    continue
                tot += p.numel()
                tr += p.numel() if p.requires_grad else 0
            if tot:
                out[label] = (tot, tr)
        return out

    def trainable_parameters(self):
        return [(n, p) for n, p in self.named_parameters() if p.requires_grad]
