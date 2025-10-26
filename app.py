"""Streamlit app for semi-automated electrophoresis gel annotation."""
from __future__ import annotations

import io
from typing import Dict, List, Optional

import cv2
import numpy as np
import streamlit as st

from gelannotator import (
    Lane,
    LaneDetectionParams,
    LaneGroup,
    build_annotation_payload,
    create_overlay_image,
    create_svg,
    detect_lanes,
    identify_ladder_lane,
)
from gelannotator.export import payload_to_json


st.set_page_config(page_title="Gel Annotator", layout="wide")
st.title("Electrophoresis Gel Annotator")


def _load_image(upload) -> Optional[np.ndarray]:
    if upload is None:
        return None
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image


def _ensure_state_keys():
    for key, default in {
        "lane_labels": {},
        "ladder_label": "Ladder",
        "lane_groups": [],
    }.items():
        if key not in st.session_state:
            st.session_state[key] = (
                default.copy() if isinstance(default, dict) else list(default)
            )


def _update_labels(lanes: List[Lane]):
    labels: Dict[int, str] = st.session_state.setdefault("lane_labels", {})
    missing = {lane.index: f"Lane {lane.index}" for lane in lanes if lane.index not in labels}
    labels.update(missing)
    st.session_state["lane_labels"] = labels


_ensure_state_keys()

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Upload gel image", type=["png", "jpg", "jpeg", "tif", "tiff"])

    st.header("Lane detection")
    top_fraction = st.slider("Top fraction for wells", 0.05, 0.5, 0.2, 0.05)
    min_distance = st.slider("Minimum lane spacing (px)", 10, 200, 40, 5)
    intensity_threshold = st.slider("Projection threshold", 0.05, 0.9, 0.3, 0.05)
    smoothing = st.slider("Smoothing kernel", 3, 101, 25, 2)
    detection_params = LaneDetectionParams(
        top_fraction=top_fraction,
        min_distance_px=min_distance,
        intensity_threshold=intensity_threshold,
        smoothing_kernel=smoothing,
    )


image = _load_image(uploaded)
if image is None:
    st.info("Upload a gel image to begin.")
    st.stop()

st.subheader("Detected lanes")
lanes = detect_lanes(image, detection_params)
if not lanes:
    st.error("No lanes detected. Adjust detection parameters and try again.")
    st.stop()

_update_labels(lanes)
labels = st.session_state["lane_labels"]

ladder_guess = identify_ladder_lane(image, lanes)
lane_indices = [lane.index for lane in lanes]
default_idx = 0
if ladder_guess in lane_indices:
    default_idx = lane_indices.index(ladder_guess) + 1
ladder_lane = st.selectbox(
    "Ladder lane",
    options=["None"] + lane_indices,
    index=default_idx,
)
ladder_lane_idx: Optional[int] = None if ladder_lane == "None" else int(ladder_lane)
ladder_label = st.text_input("Ladder label", st.session_state.get("ladder_label", "Ladder"))
st.session_state["ladder_label"] = ladder_label

columns = st.columns(2)
with columns[0]:
    st.markdown("### Lane labels")
    for lane in lanes:
        labels[lane.index] = st.text_input(
            f"Lane {lane.index}",
            value=labels.get(lane.index, f"Lane {lane.index}"),
            key=f"lane_label_{lane.index}",
        )

with columns[1]:
    st.markdown("### Group labels")
    with st.form("group_form"):
        start_lane = st.number_input("Start lane", min_value=1, max_value=len(lanes), value=1)
        end_lane = st.number_input(
            "End lane", min_value=1, max_value=len(lanes), value=len(lanes)
        )
        group_label = st.text_input("Group label", "Treatment A")
        submitted = st.form_submit_button("Add group")
    if submitted and group_label:
        lane_range = list(range(int(start_lane), int(end_lane) + 1))
        st.session_state.setdefault("lane_groups", []).append(LaneGroup(group_label, lane_range))
    for idx, group in enumerate(st.session_state.get("lane_groups", [])):
        cols = st.columns([3, 1])
        with cols[0]:
            st.write(f"{group.label}: lanes {group.lanes}")
        with cols[1]:
            if st.button("Remove", key=f"remove_group_{idx}"):
                st.session_state["lane_groups"].pop(idx)
                st.experimental_rerun()

st.session_state["lane_labels"] = labels

annotated = create_overlay_image(
    image,
    lanes,
    labels,
    ladder_lane_idx,
    ladder_label,
    st.session_state.get("lane_groups", []),
)
st.image(annotated, caption="Detected lanes with annotations", use_column_width=True)

payload = build_annotation_payload(
    lanes,
    labels,
    ladder_lane_idx,
    ladder_label,
    st.session_state.get("lane_groups", []),
)

png_buffer = io.BytesIO()
annotated.save(png_buffer, format="PNG")
png_data = png_buffer.getvalue()

svg_data = create_svg(
    image,
    lanes,
    labels,
    ladder_lane_idx,
    ladder_label,
    st.session_state.get("lane_groups", []),
)

json_data = payload_to_json(payload)

st.download_button("Download annotated PNG", png_data, file_name="annotated_gel.png")
st.download_button("Download SVG", svg_data, file_name="annotated_gel.svg")
st.download_button(
    "Download annotations JSON", json_data, file_name="annotated_gel.json"
)

st.markdown("---")
st.markdown(
    "Adjust the detection thresholds if wells are missed. Use the group section to add"
    " multi-lane annotations (e.g., treatments)."
)
