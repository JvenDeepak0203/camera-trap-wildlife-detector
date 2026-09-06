# Camera Trap Wildlife Detector

A computer vision system that detects whether a camera-trap photograph contains
an animal or is an empty/false-triggered frame. Built to address a real bottleneck
in wildlife conservation: camera traps generate huge volumes of images, most of
which are empty, and manually reviewing them all is a major time sink for
field researchers.

**Live demo:** [add your Streamlit Cloud URL here]

## How it works

Three independent models each vote on whether an image contains an animal,
combined by majority vote:

1. **ResNet18** — fine-tuned on labeled camera-trap images from
   [iWildCam 2020](https://www.kaggle.com/c/iwildcam-2020-fgvc7), with a
   Grad-CAM heatmap showing which pixels most influenced its decision.
2. **CLIP** (ViT-B-32) — scores the image against a set of candidate text
   descriptions.
3. **MegaDetector** (MDv5a) — a model purpose-built for camera-trap detection,
   drawing bounding boxes around detected animals.

Any disagreement between models is flagged explicitly rather than hidden
behind a single confidence number.

## Current results

Validation accuracy: **87–88%** (current model: v5). Full model version
history, bias analysis, and documented failure cases are in
[REPORT.md](REPORT.md).

## Running it locally

```bash
git clone https://github.com/JvenDeepak0203/camera-trap-wildlife-detector.git
cd camera-trap-wildlife-detector
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Trained weights ship in the repo — no dataset download needed just to run it.

## Known limitations (brief)

- No species-level identification, only animal presence/absence
- No true "I don't know" option for out-of-distribution input
- MegaDetector's bounding boxes are more reliable than its confidence scores on camouflaged/difficult images

See [REPORT.md](REPORT.md) for the full development history, bias analysis, and specific documented test cases.