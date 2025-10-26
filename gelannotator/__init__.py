"""Gel annotator package."""

from .lane_detection import LaneDetectionParams, Lane, detect_lanes
from .ladder_detection import identify_ladder_lane
from .overlay import (
    LaneGroup,
    TextRenderOptions,
    GridOptions,
    create_overlay_image,
    create_svg,
)
from .export import build_annotation_payload

__all__ = [
    "LaneDetectionParams",
    "Lane",
    "detect_lanes",
    "identify_ladder_lane",
    "LaneGroup",
    "create_overlay_image",
    "create_svg",
    "TextRenderOptions",
    "GridOptions",
    "build_annotation_payload",
]
