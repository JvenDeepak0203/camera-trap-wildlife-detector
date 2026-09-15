# Camera Trap Wildlife Detector

Most camera-trap photographs are empty. This tool tells you which ones aren't.

Camera traps fire on motion, and most of what they capture is vegetation moving or a
shadow shifting. Researchers review all of it by hand. This system scores each frame
on how likely it is to contain an animal, so the empty ones can be set aside.

**96.2% accuracy** on 2,961 images from 314 camera locations excluded from all
training and tuning.

---

## How it works

Two models score each image and their scores are blended:

| Model | Role | Weight |
|---|---|---|
| **ResNet18 v6-320** | Classifier fine-tuned on WCS camera-trap data | 30% |
| **MegaDetector v5a** | Purpose-built camera-trap detector | 70% |

Scores are combined as `0.3 × resnet + 0.7 × megadetector` and thresholded at 0.275.
Both the weighting and the threshold were found by sweeping every value against a
validation set, then confirmed on data neither had touched.

**CLIP** (ViT-B-32) is shown alongside as a plain-language scene description, but
takes no part in the verdict. Measured at 0.897 AUC after phrase optimization, it is
well short of the two scoring models.

Grad-CAM shows which pixels drove the classifier's decision. MegaDetector's bounding
boxes show where it found candidate regions. Disagreement between the two scoring
models is flagged rather than hidden.

---

## Results

On the held-out test split of 314 camera locations that appear nowhere in training
and were never used to tune any parameter:

| | AUC | Accuracy |
|---|---|---|
| ResNet18 v6 alone | 0.9669 | n/a |
| MegaDetector alone | 0.9687 | n/a |
| **Blended** | **0.9909** | **96.15%** |

MegaDetector alone outperforms this project's classifier. The blend beats both
because the two models fail on different images.

---

## Notable findings

**Random splits inflate camera-trap accuracy by about nine points.** An earlier
version of this project reported 87–88% on a random split. Re-measured on a
location-disjoint split, the same model scored 78.8%. Frames from one camera share
background and lighting and often arrive in bursts, so a random split lets a model
score well by recognising backgrounds.

**Detection fails sharply below 5% of frame.** Recall holds near 0.90 for animals
occupying more than 5% of the image and collapses to 0.485 below 1%. About 73% of all
missed animals are small ones. The cause is the resize step: an animal filling 1% of
a frame survives resizing as a patch smaller than a single cell of the network's final
feature map.

**Raising resolution at inference time makes things worse, not better.** Predicted to
help small animals, it instead degraded recall in every size bin, because calibration
collapses faster than discrimination improves. A paired bootstrap showed the
underlying signal was real but masked; fine-tuning at 320×320 converted it into
+0.0236 AUC on small animals, against a predicted +0.0215.

**A better classifier did not make a better ensemble.** A ResNet50 backbone was
significantly stronger alone (+0.0071 AUC, p ≈ 1.0) but produced no improvement to
the blend, because its gains were redundant with MegaDetector's existing strengths.

Full experimental history, including three negative results, is in
[REPORT.md](REPORT.md).

---

## Running it locally

```bash
git clone https://github.com/JvenDeepak0203/camera-trap-detector.git
cd camera-trap-detector
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The ResNet weights (`v6_320.pth`, 45 MB) ship with the repo. MegaDetector's weights
(280 MB) download automatically on first run.

---

## Limitations

- Animals under 5% of frame are detected substantially less reliably
- Presence/absence only, no species identification
- No "uncertain" output; every image is forced into one of two labels
- People and vehicles count as *not animals*, following the source dataset
- Unreliable on anything that isn't camera-trap imagery
- Realistic replicas and decoys are classified as animals
- Geographic coverage is that of the WCS dataset; transfer elsewhere is unverified

---

## Credits

Built by Jven Deepak.

- [MegaDetector v5a](https://github.com/agentmorris/MegaDetector) by Dan Morris and
  contributors, developed at Microsoft AI for Earth, now community-maintained
- [WCS Camera Traps](https://lila.science/datasets/wcscameratraps), 1.37M images
  contributed by the [Wildlife Conservation Society](https://www.wcs.org/), hosted by
  [LILA BC](https://lila.science/) under the Community Data License Agreement
- [OpenCLIP](https://github.com/mlfoundations/open_clip) (ViT-B-32) by LAION and
  contributors
- [iWildCam 2020](https://www.kaggle.com/c/iwildcam-2020-fgvc7), used for earlier
  model versions

A research and portfolio project, not a production conservation tool.
