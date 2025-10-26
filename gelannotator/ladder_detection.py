"""Heuristics to guess the ladder lane."""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from .lane_detection import Lane
from .lane_detection import _smooth_signal as _smooth
from .lane_detection import _find_peaks as _peaks


def _lane_profile(gray: np.ndarray, lane: Lane) -> np.ndarray:
    x1 = max(lane.x, 0)
    x2 = min(lane.x + lane.width, gray.shape[1])
    roi = gray[:, x1:x2]
    if roi.size == 0:
        return np.zeros(gray.shape[0], dtype=np.float32)
    profile = roi.sum(axis=1).astype(np.float32)
    profile = _smooth(profile, max(3, lane.width // 3))
    profile = cv2.normalize(profile, None, 0, 1.0, cv2.NORM_MINMAX)
    return profile


def identify_ladder_lane(gray: np.ndarray, lanes: List[Lane]) -> Optional[int]:
    """Return the 1-based index of the ladder lane if one can be identified."""

    if not lanes:
        return None

    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    best_idx = None
    best_score = -np.inf

    for lane in lanes:
        profile = _lane_profile(gray, lane)
        if profile.size < 10:
            continue
        threshold = 0.4 * profile.max()
        peaks = _peaks(profile, max(8, lane.height // 40), threshold)
        if len(peaks) < 3:
            continue
        diffs = np.diff(peaks)
        if len(diffs) == 0:
            continue
        spacing_mean = float(np.mean(diffs))
        spacing_std = float(np.std(diffs))
        spacing_score = spacing_mean / (1 + spacing_std)
        score = len(peaks) + spacing_score
        if score > best_score:
            best_score = score
            best_idx = lane.index

    return best_idx
