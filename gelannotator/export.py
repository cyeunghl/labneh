"""Export utilities for annotation metadata."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

from .lane_detection import Lane
from .overlay import LaneGroup


@dataclass
class AnnotationPayload:
    lanes: List[Dict[str, int | str]]
    ladder_lane: Optional[int]
    labels: Dict[str, str]
    group_labels: List[Dict[str, object]]


def build_annotation_payload(
    lanes: Iterable[Lane],
    lane_labels: Dict[int, str],
    ladder_lane: Optional[int],
    ladder_label: str,
    groups: Iterable[LaneGroup],
) -> AnnotationPayload:
    lane_entries: List[Dict[str, int | str]] = []
    labels_dict: Dict[str, str] = {}

    for lane in lanes:
        default_label = f"Lane {lane.index}"
        label = lane_labels.get(lane.index, "").strip()
        if label == default_label:
            label = ""
        if ladder_lane and lane.index == ladder_lane and not label:
            label = ladder_label or "Ladder"
        entry = {
            "index": lane.index,
            "x": lane.x,
            "label": label or default_label,
        }
        lane_entries.append(entry)
        labels_dict[str(lane.index)] = entry["label"]

    group_entries: List[Dict[str, object]] = []
    for group in groups:
        group_entries.append({"lanes": group.lanes, "label": group.label})

    return AnnotationPayload(
        lanes=lane_entries,
        ladder_lane=ladder_lane,
        labels=labels_dict,
        group_labels=group_entries,
    )


def payload_to_json(payload: AnnotationPayload) -> str:
    return json.dumps(asdict(payload), indent=2)
