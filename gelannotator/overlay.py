"""Drawing helpers for annotated overlays."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import cv2
from PIL import Image, ImageDraw, ImageFont
import svgwrite

from .lane_detection import Lane


@dataclass
class LaneGroup:
    label: str
    lanes: List[int]


_DEFAULT_FONT = None


def _get_font(size: int = 18) -> ImageFont.ImageFont:
    global _DEFAULT_FONT
    if _DEFAULT_FONT is None:
        try:
            _DEFAULT_FONT = ImageFont.truetype("DejaVuSans.ttf", size)
        except (OSError, IOError):
            _DEFAULT_FONT = ImageFont.load_default()
    return _DEFAULT_FONT


def create_overlay_image(
    image: np.ndarray,
    lanes: Iterable[Lane],
    lane_labels: Dict[int, str],
    ladder_lane: Optional[int],
    ladder_label: str,
    groups: Iterable[LaneGroup],
    alpha: float = 0.35,
) -> Image.Image:
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()

    for lane in lanes:
        color = (0, 255, 0)
        if ladder_lane and lane.index == ladder_lane:
            color = (255, 165, 0)
        cv2.rectangle(
            overlay,
            (lane.x, lane.y),
            (lane.x + lane.width, lane.y + lane.height),
            color,
            2,
        )
    blended = cv2.addWeighted(overlay, alpha, rgb, 1 - alpha, 0)
    pil_img = Image.fromarray(blended)
    draw = ImageDraw.Draw(pil_img)
    font = _get_font()

    for lane in lanes:
        default_label = f"Lane {lane.index}"
        label = lane_labels.get(lane.index, default_label)
        label = label.strip()
        if label == default_label:
            label = ""
        text = label or default_label
        if ladder_lane and lane.index == ladder_lane:
            text = label or ladder_label or "Ladder"
        text_position = (lane.x + 5, lane.y + 5)
        draw.rectangle(
            [text_position, (text_position[0] + 4 + len(text) * 7, text_position[1] + 20)],
            fill=(0, 0, 0, 140),
        )
        draw.text(text_position, text, font=font, fill=(255, 255, 255))

    height = pil_img.height
    for group in groups:
        if not group.lanes:
            continue
        xs = [lane.x for lane in lanes if lane.index in group.lanes]
        widths = [lane.width for lane in lanes if lane.index in group.lanes]
        if not xs or not widths:
            continue
        left = min(xs)
        right = max(x + w for x, w in zip(xs, widths))
        top = max(0, height // 20)
        draw.line([(left, top), (right, top)], fill=(255, 255, 0), width=3)
        draw.text((left, max(0, top - 20)), group.label, font=font, fill=(255, 255, 0))

    return pil_img


def create_svg(
    image: np.ndarray,
    lanes: Iterable[Lane],
    lane_labels: Dict[int, str],
    ladder_lane: Optional[int],
    ladder_label: str,
    groups: Iterable[LaneGroup],
) -> str:
    height, width = image.shape[:2]
    dwg = svgwrite.Drawing(size=(width, height))
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill="white"))

    for lane in lanes:
        color = "#00FF00"
        if ladder_lane and lane.index == ladder_lane:
            color = "#FFA500"
        dwg.add(
            dwg.rect(
                insert=(lane.x, lane.y),
                size=(lane.width, lane.height),
                fill="none",
                stroke=color,
                stroke_width=2,
            )
        )
        default_label = f"Lane {lane.index}"
        label = lane_labels.get(lane.index, default_label)
        label = label.strip()
        if label == default_label:
            label = ""
        text = label or default_label
        if ladder_lane and lane.index == ladder_lane:
            text = label or ladder_label or "Ladder"
        dwg.add(
            dwg.text(
                text,
                insert=(lane.x + 5, lane.y + 20),
                fill="#000000",
                font_size=16,
                font_family="DejaVu Sans, Arial, sans-serif",
            )
        )

    for group in groups:
        xs = [lane.x for lane in lanes if lane.index in group.lanes]
        widths = [lane.width for lane in lanes if lane.index in group.lanes]
        if not xs or not widths:
            continue
        left = min(xs)
        right = max(x + w for x, w in zip(xs, widths))
        top = height * 0.05
        dwg.add(
            dwg.line((left, top), (right, top), stroke="#FFFF00", stroke_width=3)
        )
        dwg.add(
            dwg.text(
                group.label,
                insert=(left, top - 10),
                fill="#FFFF00",
                font_size=18,
                font_family="DejaVu Sans, Arial, sans-serif",
            )
        )

    return dwg.tostring()
