# Development Report

Camera Trap Wildlife Detector: full development history, experiments, and findings.

---

## 1. The problem

Camera traps fire on motion. Most of what they capture is nothing: vegetation
moving, shadows shifting, a false trigger. In the WCS dataset used here, roughly
43% of frames are empty. Field researchers review all of it by hand, and that
review is a well-known bottleneck in conservation work.

The goal of this project is a tool that scores each photograph on how likely it is
to contain an animal, so the empty frames can be set aside.

---

## 2. Final system

| Component | Role | Weight in verdict |
|---|---|---|
| ResNet18 v6-320 | Binary classifier, fine-tuned on WCS | 30% |
| MegaDetector v5a | Purpose-built camera-trap detector | 70% |
| CLIP (ViT-B-32) | Scene description, interpretation only | 0% |

Scores are blended `0.3 × resnet + 0.7 × megadetector` and thresholded at **0.275**.
The ResNet output is temperature-scaled with T = 1.656.

### Held-out test performance

Evaluated on 2,961 images from **314 camera locations excluded from all training
and from every tuning decision** (weights, threshold, temperature).

```
                precision  recall  f1-score  support
empty              0.9564  0.9668    0.9616     1476
animal             0.9666  0.9562    0.9614     1485
accuracy                             0.9615     2961

blend AUC                            0.9909
resnet alone AUC                     0.9669
megadetector alone AUC               0.9687
AUC, small animals + empties         0.9802
```

**96.15% accuracy.** Notably this is *higher* than the 95.72% measured on the
validation set the weights and threshold were tuned against, which indicates the
configuration is not overfitted to that split.

Also worth stating plainly: **MegaDetector alone outperforms the ResNet alone.**
The contribution of this project's model is as a complement, not a replacement.
The blend beats both by a wide margin because the two fail on different images.

---

## 3. Model version history

All figures below are on the location-disjoint validation set (2,994 images) unless
stated otherwise.

| Version | Change | AUC | AUC (small animals) | Accuracy |
|---|---|---|---|---|
| v2 | fc-layer only, iWildCam, 3,000 images | n/a | n/a | 84%* |
| v3 | Unfroze layer4, augmentation, hard-location oversampling | n/a | n/a | 88.3%* |
| v4 | + hard negative mining (268 examples, 5× weight) | n/a | n/a | 88.4%* |
| v5 | + expanded hard negatives (1,111 examples, 8× weight) | n/a | n/a | 88%* / **78.8%** |
| **v6-320** | **WCS data, 320×320 input, location-disjoint splits** | **0.9686** | **0.9369** | **90.8%** |
| v7 | v6 + WCS-mined hard negatives (767, 3× weight) | 0.9643 | 0.9291 | 89.0% |
| v8ctrl | Small-animal rebalancing, 8,000 images | 0.9618 | 0.9286 | n/a |
| v8 | Small-animal rebalancing, 20,000 images | 0.9690 | 0.9404 | n/a |
| r50 | ResNet50 backbone, 20,000 images | 0.9757 | 0.9546 | n/a |

\* measured on a **random** split. See §4; these figures are inflated.

**v6-320 is the deployed model.** v8 was statistically tied with it; r50 was better
alone but added nothing to the ensemble (§9).

---

## 4. The random-split error

The first significant finding of this project was that its own earlier accuracy
figures were wrong.

Versions v2 through v5 were evaluated with `train_test_split(..., stratify=...)`,
a random split. Camera-trap photographs from the same camera share background,
vegetation and lighting, and frequently arrive in bursts seconds apart. A random
split places near-identical frames on both sides, so a model can score well by
recognising backgrounds rather than animals.

Re-evaluating v5 on a **location-disjoint** split, where entire cameras are held
out:

| Split | Accuracy |
|---|---|
| Random (as originally reported) | 87–88% |
| Location-disjoint | **78.8%** |

**Roughly nine points of the originally reported accuracy were camera leakage.**

All subsequent evaluation in this project uses LILA's official location-disjoint
splits (2,507 train / 313 validation / 314 test locations), with zero overlap
verified.

A second precaution: one frame per `seq_id`. WCS labels many frames at sequence
level, and burst frames are near-duplicates. Deduplicating by sequence reduced
243,513 boxed images to 48,130 usable animal frames, but removes a second leakage
path.

---

## 5. The size cliff

Using WCS bounding-box annotations, animal size was computed as box area divided
by frame area, for every boxed image, directly from metadata.

v6's predecessor (v5) evaluated by size bin:

| Animal size (fraction of frame) | n | Recall | Mean p(animal) |
|---|---|---|---|
| < 1% | 130 | **0.485** | 0.518 |
| 1–5% | 517 | 0.640 | 0.637 |
| 5–10% | 289 | 0.907 | 0.854 |
| 10–25% | 358 | 0.899 | 0.880 |
| > 25% | 282 | 0.897 | 0.879 |

Recall is flat at approximately 0.90 above 5% of frame, then collapses. Below 1%,
the model is worse than a coin flip, and mean confidence sits at 0.518. It is not
confidently wrong, it is undecided.

The flatness above 5% is what makes this a threshold rather than a trend. A gradual
decline would suggest "bigger is easier"; a flat region followed by a cliff is the
signature of information loss.

**Mechanism.** After resizing to 224×224, the frame is 50,176 pixels. An animal at
1% of frame is roughly 500 pixels, about 22×22. ResNet18's `layer4` outputs a 7×7
grid where each cell covers a 32×32 patch. **7.1% of animals in this dataset are
smaller than a single cell of the feature map being fine-tuned.** At 5% of frame the
animal spans barely 2×2 cells, and 40.6% of the dataset falls below that.

**Consequence.** 253 of 345 total false negatives (73%) come from the sub-5% bins.
The size cliff largely *is* the accuracy gap.

---

## 6. Experiment: input resolution

**Hypothesis.** If the resize destroys the signal, feeding larger images should
improve small-animal detection. `layer4` at 448 gives a 14×14 grid, so an animal at
1% of frame goes from sub-cell to roughly 2×2 cells.

### 6a. Inference-time test: hypothesis falsified

Same v5 weights, same images, three input sizes:

| Bin | 224 | 320 | 448 |
|---|---|---|---|
| < 1% | 0.485 | 0.415 | 0.369 |
| 1–5% | 0.640 | 0.660 | 0.495 |
| 5–10% | 0.907 | 0.837 | 0.633 |
| 10–25% | 0.899 | 0.872 | 0.729 |
| > 25% | 0.897 | 0.869 | 0.801 |

Recall fell at every size, including the > 25% control bin where extra pixels cannot
possibly help. Median p(animal) on true animals fell **0.928 → 0.815 → 0.614**.

**Diagnosis: train/test resolution discrepancy.** The weights were fine-tuned at 224.
At 448, every learned feature is being asked about textures and apparent object
scales it never saw. The entire probability distribution shifted toward "empty",
a global calibration collapse rather than a size-specific effect.

### 6b. Threshold-independent check: signal recovered

Accuracy comparisons across resolutions partly measure where the fixed 0.5 threshold
happens to land. AUC ignores the threshold:

| Input | AUC (all) | AUC (< 5% animals) | Median p(animal) |
|---|---|---|---|
| 224 | 0.8593 | 0.7590 | 0.928 |
| 320 | 0.8610 | **0.7804** | 0.815 |
| 448 | 0.8221 | 0.7487 | 0.614 |

The small-animal effect is real at 320 and masked by calibration collapse. Paired
bootstrap, 2,000 resamples, n = 2,065:

```
AUC<5%:  320 - 224 = +0.0215   95% CI [+0.0037, +0.0399]   P(better) = 0.990
AUC<5%:  448 - 224 = -0.0100   95% CI [-0.0321, +0.0121]   P(better) = 0.184
```

320 is a genuine improvement. 448 is inconclusive: the point estimate is negative,
but the interval crosses zero, so no claim of a confirmed reversal.

### 6c. Fine-tuning at 320: prediction confirmed

If the mechanism is correct, fine-tuning at 320 should convert the masked gain into
real performance. Identical data, seed, and schedule; only input size differs:

| Epoch | 224 AUC<5% | 320 AUC<5% |
|---|---|---|
| 1 | 0.9049 | 0.9147 |
| 2 | 0.9095 | 0.9358 |
| 3 | 0.9156 | 0.9203 |
| 4 | 0.9039 | 0.9291 |
| 5 | 0.9110 | 0.9341 |
| 6 | 0.9133 | **0.9369** |

320 leads at every epoch on both metrics. Final gap **+0.0236**, against a predicted
+0.0215.

Paired bootstrap on the fine-tuned models:

```
AUC<5%:   320 - 224 = +0.0236   95% CI [+0.0140, +0.0334]   P = 1.000
AUC all:  320 - 224 = +0.0119   95% CI [+0.0073, +0.0166]   P = 1.000
```

Both intervals exclude zero. The small-animal effect is roughly **twice** the
overall effect, concentrated exactly where the size cliff predicted it would be.

**This is the central result of the project:** a measured failure mode, a
prediction about its cause, a falsified inference-time test, a diagnosed mechanism,
and a confirmed effect under training that landed on the predicted value.

Cost: 320×320 is roughly 2× the inference compute of 224×224. The gain is not free.

---

## 7. Experiment: hard negative mining (v7), a negative result

v5's shape-based false positives (the "popcorn problem") were fixed by mining images
the model confidently misclassified and oversampling them. v6, trained on WCS
without hard negatives, regressed on this case.

767 hard negatives were mined by running v6-320 across 11,904 WCS empty frames from
**training** locations, keeping everything scored above 0.5 (6.4% false positive
rate). These were added at 3× sampling weight, reaching 22.4% of the effective
training distribution.

| | Empty recall | Animal recall | AUC | AUC<5% |
|---|---|---|---|---|
| v6-320 | 0.913 | 0.902 | 0.9686 | 0.9369 |
| v7 | **0.972** | **0.816** | 0.9643 | 0.9291 |

Empty recall rose 5.9 points; animal recall fell 8.6. For a conservation tool,
missing an animal is the costlier error.

Crucially, a threshold sweep showed v7 is worse at **every** operating point. At
t = 0.2 it matches v6's empty recall (0.9154 vs 0.9133) while missing 2.2 points more
animals. AUC confirms it independently. The hard negatives did not improve the
trade-off curve; they shifted the model along a slightly worse one.

**Likely cause.** The negatives were mined from *training* locations, cameras v6
had already seen. Those errors are probably idiosyncratic scene details rather than
generalizable failure modes. Where you mine from matters as much as how you weight.

---

## 8. Experiment: small-animal rebalancing (v8), a null result

**Hypothesis.** Since 73% of misses are small animals, over-representing them in
training should help.

Training sets were built at 69.5% small animals (against 40.6% naturally occurring):

| | AUC | AUC<5% |
|---|---|---|
| v6-320 (8k, natural composition) | 0.9686 | 0.9369 |
| v8ctrl (8k, rebalanced) | 0.9618 | 0.9286 |
| v8 (20k, rebalanced) | 0.9690 | 0.9404 |

The control isolates the variable cleanly: same size, same architecture, same
schedule as v6, only composition changed. **Rebalancing cost 0.0068 AUC and 0.0083
on the metric it was designed to improve.**

Tripling the data recovered roughly what the rebalancing lost. Paired bootstrap of
v8 against v6:

```
AUC all:  +0.0004   95% CI [-0.0028, +0.0037]   P(v8 better) = 0.595
AUC<5%:   +0.0036   95% CI [-0.0036, +0.0103]   P(v8 better) = 0.847
```

Both intervals straddle zero, so v8 is statistically indistinguishable from v6. v6 was kept,
since swapping models on a tie would mean re-deriving the temperature, ensemble
weights and threshold for no measured gain.

**Finding.** Training on a distribution that does not match the test distribution
costs accuracy even when skewed toward the hard cases. The model calibrates to a
world where 69% of animals are tiny, then is tested in one where 40% are.

---

## 9. Experiment: ResNet50 backbone, better alone but no ensemble gain

ResNet50, ImageNet-pretrained, `layer4` and `fc` unfrozen, same 20k data and
schedule. Best epoch by validation AUC:

| | AUC | AUC<5% |
|---|---|---|
| ResNet18 v6-320 | 0.9686 | 0.9369 |
| ResNet50 | **0.9757** | **0.9546** |

```
AUC all:  r50 - v6 = +0.0071   95% CI [+0.0024, +0.0122]   P = 0.999
AUC<5%:   r50 - v6 = +0.0177   95% CI [+0.0081, +0.0271]   P = 1.000
```

Both intervals exclude zero. This is a genuine improvement, and the only one found
after v6. Three attempts to fix the *data* failed; changing *model capacity*
succeeded.

But re-tuning the ensemble around ResNet50 produced **no improvement to the blend**:

| Blend | AUC |
|---|---|
| ResNet18 v6 + MegaDetector, w = 0.3 | 0.9878 |
| ResNet50 + MegaDetector, w = 0.35 | 0.9876 |

**A significantly better component made the ensemble no better.** The improvement
was in places MegaDetector already covered, so the two models' errors became more
correlated. Ensemble value comes from complementary mistakes, and a stronger
component can be worth less to an ensemble if it becomes more similar to its
partner.

ResNet50 costs roughly 3× the inference time. It was not deployed.

---

## 10. Ensemble analysis

All three models scored across the 2,994-image validation set:

| Model | AUC | Acc @ 0.5 | Empty recall | Animal recall |
|---|---|---|---|---|
| ResNet18 v6 | 0.9686 | 0.9075 | 0.9133 | 0.9023 |
| MegaDetector v5a | 0.9633 | 0.9399 | 0.9880 | 0.8966 |
| CLIP (original phrases) | 0.7361 | 0.5478 | 0.0628 | 0.9841 |
| CLIP (revised phrases) | 0.8770 | 0.5922 | 0.1601 | 0.9810 |

Combination rules:

| Rule | AUC | Accuracy |
|---|---|---|
| Majority vote (3 models) | n/a | 0.9372 |
| ResNet + CLIP, averaged | 0.9656 | 0.8367 |
| All three, averaged | 0.9849 | 0.9409 |
| **ResNet + MegaDetector, averaged** | **0.9869** | 0.9372 |

**Adding CLIP as a third voter made the ensemble worse** (0.9849 vs 0.9869).
Averaging does not launder a weak model's opinion; it dilutes the strong ones.
Majority voting also underperformed averaging, which preserves confidence
information a hard vote discards.

### Weight and threshold

| w (ResNet share) | AUC |
|---|---|
| 0.0 (MegaDetector only) | 0.9633 |
| 0.1 | 0.9877 |
| 0.3 | 0.9878 |
| 0.5 | 0.9869 |
| 1.0 (ResNet only) | 0.9686 |

The jump from w = 0.0 to w = 0.1 is +0.024 AUC. Adding just 10% of the ResNet to
MegaDetector produces nearly the entire ensemble gain. The curve is flat from 0.1 to
0.4, so w = 0.3 was chosen rather than the exact argmax, to avoid overfitting a flat
optimum.

Threshold sweep at w = 0.3 found 0.275 optimal (95.72% accuracy, balanced recall
0.9579) against 94.32% at the intuitive 0.5.

**Why the threshold is so low:** MegaDetector returns exactly 0.0 on 49.8% of frames.
Those frames score at most `0.3 × p_resnet`, capped at 0.3. Any threshold above 0.3
would discard the ResNet's opinion entirely on half the dataset. The optimum sitting
just below that cliff is not a coincidence.

---

## 11. CLIP phrase optimization

CLIP's original phrase list was 7 animal phrases against 4 non-animal, softmaxed
together. An indifferent CLIP therefore scores about 0.64, above the 0.5 threshold,
and votes "animal" by default. Empty recall was 6.3%.

A greedy forward search over 32 candidate phrases (16 animal, 16 empty):

| Phrase set | AUC | Empty recall | Animal recall |
|---|---|---|---|
| Original (7a/4e) | 0.7361 | 0.063 | 0.984 |
| Revised (8a/8e) | 0.8770 | 0.160 | 0.981 |
| All candidates (16a/16e) | **0.7419** | 0.538 | 0.758 |
| Animal-heavy (16a/8e) | 0.8743 | 0.048 | 0.990 |
| Empty-heavy (8a/16e) | 0.7610 | 0.776 | 0.639 |
| **Greedy-optimal (10a/5e)** | **0.8967** | n/a | n/a |

Two findings:

**More phrases is not better.** Using all 32 scored 0.7419, far worse than a curated
16. Adding phrases dilutes the softmax across more categories and blurs the signal.

**Balance controls calibration, not discrimination.** Animal-heavy and empty-heavy
sets have wildly different recall profiles (4.8% vs 77.6% empty recall) but similar
AUC. The ratio determines where the effective threshold sits; it barely affects
ranking quality.

CLIP's measured ceiling is **0.8967**, against 0.969 and 0.963 for the two scoring
models. It takes no part in the verdict. CLIP was trained on internet photographs
with captions, not wide-angle infrared wildlife frames, and no amount of prompt
engineering closes that gap.

---

## 12. Documented test cases

| Image | Combined | ResNet | MegaDetector | Note |
|---|---|---|---|---|
| Leaf-mimic geckos (macro) | n/a | 97.3% | no detection | Classification and detection diverge: MegaDetector must resolve an object boundary, which leaf-mimic camouflage defeats. Out-of-distribution for both. |
| Silver pheasants (camera trap) | high | 100% | 95.1% | Both agree. Real camera-trap imagery, in-distribution. |
| Clouded leopard (camera trap) | 45.7% | **4.5%** | 63.4% | The ensemble rescuing a ResNet miss. Small, camouflaged animal, exactly the failure mode §5 predicts. |
| Hazy landscape with two people | 15.8% | 52.5% | 0.0% | Correctly empty: people are labelled not-animal. ResNet uncertain, MegaDetector decisive. |
| Popcorn (synthetic OOD probe) | n/a | 72.8% | ~0% | v5 scored 7.8% here after hard-negative mining; v6 regressed. Not camera-trap data; see §7. |

---

## 13. Known limitations

- **Small animals.** Recall falls from ~0.90 above 5% of frame to 0.485 below 1%.
  73% of all misses are small animals. Mechanism in §5.
- **No species identification.** Presence/absence only.
- **No abstain option.** Every image is forced into one of two labels.
- **People and vehicles are labelled not-animal**, following the source dataset.
- **Out-of-distribution imagery** (macro photography, landscapes, studio images,
  stylised images) gives unreliable results.
- **Realistic replicas** such as decoys, taxidermy and mechanical models are
  classified as animals. The system reasons from pixels and has no access to physical context.
- **Geographic coverage** is that of the WCS dataset: 12 countries, largely South
  American, African and Asian sites. Transfer elsewhere is unverified.
- **The popcorn regression** (§7) is unresolved. The fix that worked for v5 came
  from a different dataset, and WCS-mined replacements made the model worse overall.
  Documented rather than papered over.

---

## 14. Data and tooling

**WCS Camera Traps**, via [LILA BC](https://lila.science/datasets/wcscameratraps):
1,369,991 images, 675 species, 12 countries, roughly 50% empty. Contributed by the
Wildlife Conservation Society, released under the Community Data License Agreement.
Bounding boxes from `wcs_20220205_bboxes_with_classes`; splits from `wcs_splits.json`.

Label handling: category 0 is empty; `human` (75), `start` (347), `end` (348),
`unknown` (79) and `unidentifiable` (290) were excluded; `motorcycle` (558) was kept
and labelled not-animal, since a vehicle triggering a camera is a realistic negative.
Only annotations with `sequence_level_annotation: False` were trusted for size
analysis, since sequence-level labels propagate across bursts and are unreliable
per-frame.

**MegaDetector v5a** by Dan Morris and contributors, developed at Microsoft AI for
Earth, now community-maintained.

**OpenCLIP** ViT-B-32 by LAION and contributors.

**iWildCam 2020**, used for v2 through v5, via the `qitvision/iwildcam2020-256`
Kaggle mirror. Note this mirror is pre-shrunk to 256px, meaning training images were
downsized twice.

Training on Google Colab (Tesla T4). Deployment via Streamlit.

---

## 15. Summary of findings

1. Random splits inflate camera-trap accuracy by roughly nine points. Location-
   disjoint evaluation is not optional for this data.
2. Detection accuracy has a sharp threshold at ~5% of frame, driven by the resize
   step destroying sub-feature-map-cell detail. 73% of misses are small animals.
3. Raising input resolution at inference time *fails*, because calibration collapses
   faster than discrimination improves. The underlying gain is real but must be realised
   through fine-tuning at the target resolution. Predicted +0.0215, observed +0.0236.
4. Hard negative mining from training locations degraded performance at every
   operating point.
5. Oversampling the failure mode degraded performance on the failure mode.
6. Model capacity, not data volume or composition, was the binding constraint on the
   classifier.
7. A significantly stronger classifier produced no ensemble improvement, because its
   gains were redundant with its partner's strengths.
8. For CLIP, phrase count is not the lever; phrase balance controls calibration and
   phrase relevance controls discrimination. Its ceiling on this task is ~0.90 AUC.

Of eight experiments run after v6, two produced confirmed improvements and the rest
were negative or null. All are reported.
