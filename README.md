# Camera Trap Wildlife Detector

A computer vision system that detects whether a camera-trap photograph contains
an animal or is an empty/false-triggered frame. Built to address a real bottleneck
in wildlife conservation: camera traps generate huge volumes of images, most of
which are empty, and manually reviewing them all is a major time sink for
field researchers.

**Live demo:** https://camera-trap-detector.streamlit.app/

## How it works

Three independent models each vote on whether an image contains an animal:

1. **ResNet18** — fine-tuned from ImageNet weights on labeled camera-trap images
   from the [iWildCam 2020](https://www.kaggle.com/c/iwildcam-2020-fgvc7) dataset.
2. **CLIP** (ViT-B-32) — a general-purpose vision-language model, used here to
   score the image against a set of candidate text descriptions (e.g. "a wide
   shot of an animal's full body," "a photo of an inanimate object").
3. **MegaDetector** — a model purpose-built for camera-trap animal/person/vehicle
   detection by conservation-tech researchers; the closest thing to an industry
   standard tool for this exact task.

The final verdict is decided by **majority vote** across the three models, with
any disagreement flagged explicitly to the user rather than hidden behind a
single confidence number.

A **Grad-CAM heatmap** is also shown for the ResNet model, visualizing which
pixels most influenced its decision (not a claim that the model "knows"
concepts like eyes or fur — just which regions drove the output).

## Model version history

| Version | Change | Overall accuracy | Animal recall | Empty recall |
|---|---|---|---|---|
| v2 | Baseline: fc-layer-only fine-tuning, 3,000 images | 84% | 90% | 75% |
| v3 | Unfroze layer4, added augmentation, oversampled hard camera-trap locations | 88.3% | 87.8% | 88.9% |
| v4 | + hard negative mining (268 examples), 5x oversampled | 88.4% | 89.5% | 86.8% |
| v5 (current) | + expanded hard negative mining (1,106 examples), 8x oversampled | 87.0% | 83.4% | 92.8% |

Each version targeted a specific, measured problem found in the previous one
(see Known Limitations & Findings below) rather than chasing a single accuracy
number.

## Known limitations & findings

**Location-based bias (found after v2, addressed in v3):** accuracy on the
original model varied enormously by camera-trap site — from 52.9% at the
worst-performing location to 100% at the best, against an overall accuracy of
82.2%. Visual inspection of the worst site showed the model struggled
specifically with animals camouflaged against dense, textured vegetation. After
v3 (deeper fine-tuning + oversampling known-hard locations), the worst site
improved to 66.7% and the overall spread narrowed substantially.

**Overconfidence (found after v3):** v3 showed signs of overfitting past
~epoch 10 (validation accuracy plateaued while training loss kept falling),
resulting in unrealistic 100.0% confidence scores even on ambiguous images.
Fixed using **temperature scaling**, a standard post-hoc calibration technique
that rescales model outputs to be honestly uncertain without changing what the
model predicts.

**Shape-based false positives ("the popcorn problem"):** v3's deeper
fine-tuning improved camouflage handling but made it more prone to false
positives on blob-shaped objects that resemble animals only in silhouette — a
photo of a piece of popcorn was classified "animal" at 98.1% confidence.
Addressed through **hard negative mining**: running the model against a fresh
pool of real labeled-empty images, finding the ones it confidently
misclassified (8-10% of a fresh 3,000–8,000 image sample), and oversampling
those specific examples in retraining. This reduced the popcorn false-positive
confidence from 98.1% to 7.8% (ResNet alone) — the ensemble's majority vote now
correctly labels this case "Empty frame."

**A real trade-off, not hidden:** the hard-negative-mining fixes in v4/v5 came
at a cost — animal recall dropped from v3's camouflage-focused peak (89.5% in
v4) to 83.4% in v5, as the model became more broadly skeptical of "animal-shaped"
patterns. Since missing a real animal (false negative) is arguably the more
costly error for this tool's intended use, this is a genuine, documented
trade-off rather than a clean win, and worth further tuning if this were
deployed for real use.

**No true "I don't know" option:** given a completely out-of-distribution
input (tested with a cartoon vector illustration), the ensemble is still forced
to pick one of two labels. It handled this test case correctly (all three
models agreed "empty"), but there's no confidence floor below which the system
says "I can't tell" rather than guessing.

**Model doesn't identify species** — only whether an animal is present at all.
Extending to species-level classification would use the same training
approach, with more output classes and (likely) more data per class.

## Bias analysis

Accuracy was checked across 62 individual camera-trap locations (those with
15+ validation images, for statistical reliability). Results showed a gradual,
structural spread rather than isolated outliers — evidence that the model was
partly learning site-specific visual context (background, lighting, camera
angle) rather than purely animal-presence features. Full per-location numbers
and the retraining response are documented above.

## Running it locally

```bash
git clone https://github.com/JvenDeepak0203/camera-trap-wildlife-detector.git
cd camera-trap-wildlife-detector
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open the printed local URL in your browser. The trained weights
(`wildlife_detector_v5.pth`) ship in the repo, so no retraining or dataset
download is needed just to run it.

## Dataset

[iWildCam 2020](https://www.kaggle.com/c/iwildcam-2020-fgvc7), via the
`qitvision/iwildcam2020-256` Kaggle mirror (pre-resized to 256px). Trained on
a sample of ~15,000 labeled images plus mined hard negatives, evaluated on a
held-out validation split never seen during training.