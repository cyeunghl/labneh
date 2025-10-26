"""Lane detection utilities for electrophoresis gel images."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class Lane:
    """Representation of a detected lane."""

    index: int
    x: int
    y: int
    width: int
    height: int

    @property
    def center_x(self) -> int:
        return self.x + self.width // 2


@dataclass
class LaneDetectionParams:
    """Parameters controlling lane detection."""

    top_fraction: float = 0.2
    min_distance_px: int = 40
    intensity_threshold: float = 0.3
    smoothing_kernel: int | None = None


def _smooth_signal(signal: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = max(1, kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = cv2.GaussianBlur(signal.reshape(-1, 1), (0, 0), sigmaX=kernel_size / 3)
    return smoothed.reshape(-1)


def _find_peaks(signal: np.ndarray, min_distance: int, threshold: float) -> List[int]:
    peaks: List[int] = []
    last_peak_idx = -min_distance * 2
    last_peak_value = -np.inf
    last_peak_position = -1

    for idx in range(1, len(signal) - 1):
        value = signal[idx]
        if value <= threshold:
            continue
        if value > signal[idx - 1] and value >= signal[idx + 1]:
            if idx - last_peak_idx >= min_distance:
                peaks.append(idx)
                last_peak_idx = idx
                last_peak_value = value
                last_peak_position = idx
            else:
                if value > last_peak_value:
                    peaks[-1] = idx
                    last_peak_idx = idx
                    last_peak_value = value
                    last_peak_position = idx
    return peaks


def detect_lanes(image: np.ndarray, params: LaneDetectionParams) -> List[Lane]:
    """Detect lanes in a gel image using simple projection profiling."""

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    height, width = gray.shape
    top_rows = max(1, int(height * params.top_fraction))
    top_region = gray[:top_rows, :]

    projection = top_region.sum(axis=0).astype(np.float32)

    if params.smoothing_kernel:
        projection = _smooth_signal(projection, params.smoothing_kernel)
    else:
        kernel_estimate = max(3, int(width * 0.01))
        projection = _smooth_signal(projection, kernel_estimate)

    max_val = float(projection.max())
    threshold = max_val * params.intensity_threshold

    peaks = _find_peaks(projection, params.min_distance_px, threshold)

    if not peaks:
        return []

    lane_widths = []
    for idx, center in enumerate(peaks):
        if idx == 0:
            left = 0
        else:
            left = (peaks[idx - 1] + center) // 2
        if idx == len(peaks) - 1:
            right = width
        else:
            right = (center + peaks[idx + 1]) // 2
        lane_widths.append((left, right))

    lanes: List[Lane] = []
    for idx, ((left, right), center) in enumerate(zip(lane_widths, peaks)):
        lane_width = max(1, right - left)
        lane = Lane(
            index=idx + 1,
            x=int(max(0, left)),
            y=0,
            width=int(min(width - left, lane_width)),
            height=height,
        )
        lanes.append(lane)
    return lanes
