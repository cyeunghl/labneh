# labneh
Semi-automated annotator for electrophoresis gel images.

## Features

- Detects gel lanes from uploaded PNG/JPEG/TIFF files using OpenCV projection profiling.
- Heuristically identifies the molecular weight ladder lane.
- Streamlit UI for editing lane labels, renaming the ladder, and managing multi-lane groups.
- Adjustable label styling (font choice, size, rotation) with live detection preview controls.
- Optional image inversion plus configurable grid overlays for preview and export workflows.
- Exports an annotated PNG, scalable SVG, and structured JSON metadata.

## Getting started

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Launch the Streamlit app:

   ```bash
   streamlit run app.py
   ```

3. Upload a gel image, tune detection parameters if necessary, and adjust labels/groups.

4. Download the annotated assets for figure preparation or downstream analysis.
