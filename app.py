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
    GridOptions,
    TextRenderOptions,
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
    invert_image = st.checkbox("Invert image", value=False, help="Invert intensities before detection")

    st.header("Lane detection")
    top_fraction = st.slider("Top fraction for wells", 0.05, 0.5, 0.2, 0.05)
    min_distance = st.slider("Minimum lane spacing (px)", 10, 200, 40, 5)
    intensity_threshold = st.slider("Projection threshold", 0.05, 0.9, 0.3, 0.05)
    smoothing = st.slider("Smoothing kernel", 3, 101, 25, 2)
    show_preview = st.checkbox("Live lane preview", value=True)
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

if invert_image:
    image = cv2.bitwise_not(image)

lanes = detect_lanes(image, detection_params)
if not lanes:
    st.error("No lanes detected. Adjust detection parameters and try again.")
    st.stop()

st.subheader("Detected lanes")
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

with st.expander("Label styling", expanded=False):
    font_family = st.selectbox(
        "Font",
        options=["Arial", "DejaVu Sans", "Liberation Sans", "System Default"],
        index=0,
    )
    font_size = st.slider("Font size", min_value=8, max_value=64, value=18)
    rotation = st.slider("Text rotation", min_value=0, max_value=90, value=0)

text_options = TextRenderOptions(
    font_family=font_family,
    font_size=font_size,
    rotation=rotation,
    show_text=True,
)

with st.expander("Grid overlay", expanded=False):
    grid_preview = st.checkbox("Show grid on preview", value=False)
    grid_rows = st.number_input(
        "Grid rows",
        min_value=0,
        value=0,
        help="0 disables horizontal grid lines",
    )
    grid_cols = st.number_input(
        "Grid columns",
        min_value=0,
        value=len(lanes),
        help="0 disables vertical grid lines",
    )
    grid_color_hex = st.color_picker("Grid color", "#FFFFFF")
    grid_stroke = st.slider("Grid line width", min_value=1, max_value=10, value=1)
    include_grid_exports = st.checkbox(
        "Include grid in downloads",
        value=grid_preview,
        help="Export PNG/SVG with grid overlay",
    )


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


grid_preview_options = GridOptions(
    enabled=grid_preview,
    rows=int(grid_rows),
    columns=int(grid_cols),
    color=_hex_to_rgb(grid_color_hex),
    stroke_px=int(grid_stroke),
)

grid_export_options = GridOptions(
    enabled=include_grid_exports,
    rows=int(grid_rows),
    columns=int(grid_cols),
    color=_hex_to_rgb(grid_color_hex),
    stroke_px=int(grid_stroke),
)

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

if show_preview:
    preview_image = create_overlay_image(
        image,
        lanes,
        labels,
        ladder_lane_idx,
        ladder_label,
        st.session_state.get("lane_groups", []),
        text_options=TextRenderOptions(show_text=False),
        grid_options=grid_preview_options,
    )
    st.image(
        preview_image,
        caption="Lane detection preview",
        use_container_width=True,
    )

annotated = create_overlay_image(
    image,
    lanes,
    labels,
    ladder_lane_idx,
    ladder_label,
    st.session_state.get("lane_groups", []),
    text_options=text_options,
    grid_options=grid_export_options,
)
st.image(annotated, caption="Detected lanes with annotations", use_container_width=True)

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
    text_options=text_options,
    grid_options=grid_export_options,
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
