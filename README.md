# Camera Trap Wildlife Detector

Most camera-trap photographs are empty. This tells you which ones aren't.

**Live demo:** https://camera-trap-wildlife-detector.streamlit.app

Camera traps fire whenever something moves, so most of what they record is grass
blowing or a shadow shifting. Someone still has to look through all of it. This
scores every frame on how likely it is to have an animal in it, so the empty ones
can be skipped.

It gets that call right **96.2%** of the time on cameras it has never seen.

---

## How it works

I use two models and blend their scores:

| Model | What it does | Share of the verdict |
|---|---|---|
| **ResNet18 v6-320** | A classifier I fine-tuned on WCS camera-trap images | 30% |
| **MegaDetector v5a** | A detector built specifically for camera traps | 70% |

The final score is `0.3 × resnet + 0.7 × megadetector`, and anything above 0.275
counts as an animal. I found both the weighting and the cutoff by sweeping every
value against a validation set, then checked them on data neither had seen.

**CLIP** is also in the app, matching each photo against plain-English descriptions,
but it doesn't vote. I tested it properly and it isn't good enough at this: 0.897
AUC even after optimising the phrase list, against 0.969 and 0.963 for the other two.

The app also shows a Grad-CAM heatmap of what the classifier was looking at,
MegaDetector's bounding boxes, and a warning when the two scoring models disagree.

---

## How well it actually works

Tested on 2,961 images from 314 camera locations that were kept out of training and
out of every tuning decision:

| | AUC | Accuracy |
|---|---|---|
| ResNet18 v6 on its own | 0.9669 | n/a |
| MegaDetector on its own | 0.9687 | n/a |
| **Both blended** | **0.9909** | **96.15%** |

Accuracy is only listed for the blend because it's the only one with a threshold
tuned for it. MegaDetector on its own actually scores slightly higher than my model
does. The point isn't that mine is better; it's that the two fail on different
images, so putting them together beats either one.

---

## Things I found along the way

**My earlier accuracy numbers were wrong.** I'd been reporting 87-88%, using a
random train/test split. But photos from the same camera share the same background
and often come in bursts a few seconds apart, so a random split puts near-identical
frames on both sides and the model can score well just by recognising backgrounds.
When I split by camera location instead, the same model got **78.8%**. Nine points
of my reported accuracy was leakage.

**There's a sharp cliff at 5% of frame.** Recall sits around 0.90 for animals taking
up more than 5% of the image and falls off a cliff to 0.485 below 1%. About 73% of
everything the model misses is a small animal. The reason is the resize step: an
animal filling 1% of a frame ends up as a patch smaller than one cell of the
network's final feature map, so the detail is gone before the model sees it.

**Bigger input images made things worse, not better.** I expected higher resolution
to help with small animals. It didn't. Recall dropped in every size bin, including
large animals where extra pixels can't possibly matter. What was happening was
calibration collapse: the model had been trained at one resolution and got confused
at another. A bootstrap showed the small-animal gain was real but buried, and
retraining at 320x320 brought it out: +0.0236 AUC, against +0.0215 predicted.

**A better model didn't make a better ensemble.** ResNet50 beat ResNet18 clearly on
its own. But blending it with MegaDetector gave 0.9876 against 0.9878 for the
smaller model, so no improvement at all. Its gains were in places MegaDetector
already handled well, and an ensemble only benefits when its parts fail differently.

Everything I tried, including three experiments that didn't work, is written up in
[REPORT.md](REPORT.md).

---

## Running it yourself

```bash
git clone https://github.com/JvenDeepak0203/camera-trap-detector.git
cd camera-trap-detector
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
streamlit run streamlit_app.py
```

My weights (`v6_320.pth`, 45 MB) are in the repo. MegaDetector's are 280 MB and
download on first run, so the first launch is slow.

---

## What it can't do

- Small animals, under about 5% of the frame, are much less reliable
- It only says animal or empty, it doesn't identify species
- There's no "not sure" option, every image gets forced into one of the two
- People and vehicles count as *not animals*, which is how the source data labels them
- Anything that isn't a camera-trap photo (macro shots, landscapes, studio images)
  gives unreliable results
- A realistic decoy or a taxidermy mount will be called an animal
- It's only been tested on WCS data, so I can't say how it does elsewhere

---

## Credits

Built by Jven Deepak.

This leans heavily on other people's work:

- [MegaDetector v5a](https://github.com/agentmorris/MegaDetector) by Dan Morris and
  contributors, built at Microsoft AI for Earth and now community-maintained
- [WCS Camera Traps](https://lila.science/datasets/wcscameratraps), 1.37M images
  contributed by the [Wildlife Conservation Society](https://www.wcs.org/) and hosted
  by [LILA BC](https://lila.science/) under the Community Data License Agreement
- [OpenCLIP](https://github.com/mlfoundations/open_clip) (ViT-B-32) by LAION and
  contributors
- [iWildCam 2020](https://www.kaggle.com/c/iwildcam-2020-fgvc7), which earlier
  versions of my model were trained on

This is a portfolio and research project, not something a conservation organization should
depend on.
