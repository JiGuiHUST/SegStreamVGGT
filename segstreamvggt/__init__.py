# SegStreamVGGT: query-based semantic segmentation for aerial datasets,
# built on StreamVGGT and imitating the structure of SegVGGT.

__version__ = "0.1.0"

from .models.segstreamvggt import SegStreamVGGT, combine_query_predictions

__all__ = ["SegStreamVGGT", "combine_query_predictions"]
