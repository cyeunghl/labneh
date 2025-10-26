"""Drawing helpers for annotated overlays."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import svgwrite

from .lane_detection import Lane


@dataclass
class LaneGroup:
    label: str
    lanes: List[int]


@dataclass
class TextRenderOptions:
    font_family: str = "Arial"
    font_size: int = 18
    rotation: int = 0
    show_text: bool = True


@dataclass
class GridOptions:
    enabled: bool = False
    rows: int = 0
    columns: int = 0
    color: Tuple[int, int, int] = field(default_factory=lambda: (255, 255, 255))
    stroke_px: int = 1


@lru_cache(maxsize=32)
def _load_font(font_family: str, font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = {
        "arial": ["Arial.ttf", "arial.ttf", "DejaVuSans.ttf"],
        "dejavu sans": ["DejaVuSans.ttf", "Arial.ttf", "arial.ttf"],
        "liberation sans": ["LiberationSans-Regular.ttf", "DejaVuSans.ttf", "Arial.ttf"],
    }
    font_files = candidates.get(font_family.lower(), []) + [font_family]
    for candidate in font_files:
        try:
            return ImageFont.truetype(candidate, font_size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _draw_rotated_text(
    base: Image.Image,
    text: str,
    position: Tuple[int, int],
    font: ImageFont.ImageFont,
    rotation: int,
    fill: Tuple[int, int, int] = (255, 255, 255),
    background: Optional[Tuple[int, int, int, int]] = None,
) -> None:
    if not text:
        return
    if hasattr(font, "getbbox"):
        text_bbox = font.getbbox(text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    else:
        text_width, text_height = font.getsize(text)
    text_image = Image.new("RGBA", (text_width + 8, text_height + 8), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_image)
    if background is not None:
        draw.rectangle(
            [
                (0, 0),
                (text_width + 8, text_height + 8),
            ],
            fill=background,
        )
    draw.text((4, 4), text, font=font, fill=fill)
    if rotation:
        text_image = text_image.rotate(rotation, expand=True)
    base.paste(text_image, position, text_image)


def create_overlay_image(
    image: np.ndarray,
    lanes: Iterable[Lane],
    lane_labels: Dict[int, str],
    ladder_lane: Optional[int],
    ladder_label: str,
    groups: Iterable[LaneGroup],
    alpha: float = 0.35,
    text_options: Optional[TextRenderOptions] = None,
    grid_options: Optional[GridOptions] = None,
) -> Image.Image:
    text_opts = text_options or TextRenderOptions()
    grid_opts = grid_options or GridOptions()
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
    pil_img = Image.fromarray(blended).convert("RGBA")
    font = _load_font(text_opts.font_family, text_opts.font_size)
    draw = ImageDraw.Draw(pil_img)

    if grid_opts.enabled and (grid_opts.rows > 0 or grid_opts.columns > 0):
        height, width = pil_img.height, pil_img.width
        if grid_opts.rows > 0:
            for row in range(1, grid_opts.rows):
                y = int(height * row / grid_opts.rows)
                draw.line(
                    [(0, y), (width, y)],
                    fill=grid_opts.color,
                    width=grid_opts.stroke_px,
                )
        if grid_opts.columns > 0:
            for col in range(1, grid_opts.columns):
                x = int(width * col / grid_opts.columns)
                draw.line(
                    [(x, 0), (x, height)],
                    fill=grid_opts.color,
                    width=grid_opts.stroke_px,
                )

    if text_opts.show_text:
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
            _draw_rotated_text(
                pil_img,
                text,
                text_position,
                font,
                rotation=max(0, min(90, text_opts.rotation)),
                fill=(255, 255, 255),
                background=(0, 0, 0, 180),
            )

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

    return pil_img.convert("RGB")


def create_svg(
    image: np.ndarray,
    lanes: Iterable[Lane],
    lane_labels: Dict[int, str],
    ladder_lane: Optional[int],
    ladder_label: str,
    groups: Iterable[LaneGroup],
    text_options: Optional[TextRenderOptions] = None,
    grid_options: Optional[GridOptions] = None,
) -> str:
    text_opts = text_options or TextRenderOptions()
    grid_opts = grid_options or GridOptions()
    height, width = image.shape[:2]
    dwg = svgwrite.Drawing(size=(width, height))
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill="white"))

    if grid_opts.enabled and (grid_opts.rows > 0 or grid_opts.columns > 0):
        if grid_opts.rows > 0:
            for row in range(1, grid_opts.rows):
                y = height * row / grid_opts.rows
                dwg.add(
                    dwg.line(
                        start=(0, y),
                        end=(width, y),
                        stroke=svgwrite.rgb(*grid_opts.color),
                        stroke_width=grid_opts.stroke_px,
                    )
                )
        if grid_opts.columns > 0:
            for col in range(1, grid_opts.columns):
                x = width * col / grid_opts.columns
                dwg.add(
                    dwg.line(
                        start=(x, 0),
                        end=(x, height),
                        stroke=svgwrite.rgb(*grid_opts.color),
                        stroke_width=grid_opts.stroke_px,
                    )
                )

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
        if text_opts.show_text:
            default_label = f"Lane {lane.index}"
            label = lane_labels.get(lane.index, default_label)
            label = label.strip()
            if label == default_label:
                label = ""
            text = label or default_label
            if ladder_lane and lane.index == ladder_lane:
                text = label or ladder_label or "Ladder"
            text_element = dwg.text(
                text,
                insert=(lane.x + 5, lane.y + text_opts.font_size + 2),
                fill="#000000",
                font_size=text_opts.font_size,
                font_family=f"{text_opts.font_family}, Arial, sans-serif",
            )
            rotation = max(0, min(90, text_opts.rotation))
            if rotation:
                text_element.rotate(
                    rotation,
                    center=(lane.x + 5, lane.y + text_opts.font_size + 2),
                )
            dwg.add(text_element)

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
