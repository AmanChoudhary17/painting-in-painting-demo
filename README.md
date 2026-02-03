# Painting in a Painting – Beginner Computer Vision Demo

This repository contains a small proof-of-concept project inspired by the
Human-AI Foundation / CERN project **"Painting in a Painting"**.

The goal is to explore whether simple computer vision techniques can detect
the presence of a hidden image beneath a visible painting.

---

## Approach

1. A visible image is treated as the top painting layer.
2. A second image is blended underneath to simulate a hidden painting.
3. The composite image is analyzed using:
   - Edge detection
   - Pixel-wise difference maps
   - Heatmap visualization

The system focuses on **detection and differentiation**, not full reconstruction.

---

## Results

The generated difference maps and heatmaps highlight regions where the
composite image deviates from the visible painting. These deviations indicate
possible underlying structure, which aligns with early-stage non-invasive
art analysis techniques.

> The project intentionally demonstrates the limitations of simple methods
> and serves as a baseline for more advanced approaches.

---

## Tools Used

- Python
- OpenCV
- NumPy
- Matplotlib

---

## How to Run

```bash
pip install opencv-python numpy matplotlib
python hidden_image_demo.py
