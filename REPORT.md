\# Development Report: Camera Trap Wildlife Detector



\## Project goal



Build a tool to help conservation researchers filter camera-trap images by

whether they contain an animal, reducing the manual review burden of mostly-empty

footage.



\## Architecture



Three independent models, combined by majority vote:

1\. ResNet18, fine-tuned on iWildCam 2020

2\. CLIP (ViT-B-32), scoring the image against candidate phrases

3\. MegaDetector (MDv5a), purpose-built for camera-trap detection



\## Model version history



| Version | Change | Overall accuracy | Animal recall | Empty recall |

|---|---|---|---|---|

| v2 | Baseline: fc-layer-only fine-tuning, 3,000 images | 84% | 90% | 75% |

| v3 | Unfroze layer4, added augmentation, oversampled hard camera-trap locations | 88.3% | 87.8% | 88.9% |

| v4 | + hard negative mining (268 examples), 5x oversampled | 88.4% | 89.5% | 86.8% |

| v5 (current) | + expanded hard negative mining (1,106 examples), 8x oversampled | 87.0% | 83.4% | 92.8% |



\## Bias analysis



Accuracy was checked across 62 camera-trap locations (15+ validation images

each). Results showed a gradual, structural spread — from 52.9% at the

worst-performing site to 100% at the best (v2), against an overall accuracy of

82.2%. Visual inspection showed the model struggled specifically with animals

camouflaged against dense, textured vegetation. After v3, the worst site

improved to 66.7%, and the spread narrowed substantially.



\## Documented failure modes \& fixes



\*\*Overconfidence:\*\* v3 showed signs of overfitting past \~epoch 10, producing

unrealistic 100.0% confidence even on ambiguous images. Fixed with

\*\*temperature scaling\*\*, a standard calibration technique.



\*\*Shape-based false positives ("the popcorn problem"):\*\* v3's deeper

fine-tuning made it prone to false positives on blob-shaped objects resembling

animals only in silhouette — a photo of popcorn scored 98.1% "animal."

Addressed via \*\*hard negative mining\*\*: finding images the model confidently

misclassified, then oversampling them in retraining. Reduced the popcorn

false-positive confidence from 98.1% to 7.8% (ResNet alone); the ensemble's

majority vote correctly labels this case "Empty frame."



\*\*A real trade-off:\*\* hard-negative-mining fixes (v4/v5) reduced animal

recall from v3's peak (89.5% in v4) to 83.4% (v5), since the model became more

broadly skeptical of "animal-shaped" patterns generally. Since missing a real

animal is the costlier error for this tool's use case, this is a genuine,

documented trade-off, not a clean win.



\## Test cases



| Test image | Result | Notes |

|---|---|---|

| Lion, clear close-up photo | Animal detected, \~95%, all agree | Handles the easy case correctly |

| Walking lioness, wide shot | Animal detected, \~94%, all agree | Required expanding CLIP's candidate phrases beyond close-up framing |

| Popcorn (shape-based false positive) | Empty frame, \~59%, flagged disagreement | Ensemble catches a real ResNet blind spot after hard negative mining |

| Cartoon potato (out-of-distribution) | Empty frame, \~95%, all agree | Handles a stylized, non-photographic input correctly |

| Camouflaged wild cat in leaf litter | Animal detected, \~80%, all agree | Each model individually only moderately confident; majority vote still correct — demonstrates the value of ensembling on genuinely hard cases |



\## Known limitations



\- No species-level classification, only presence/absence

\- No true "I don't know" option for out-of-distribution input — the ensemble

&#x20; is always forced to pick one of two labels

\- MegaDetector's bounding-box localization is often correct even when its own

&#x20; confidence score is only moderate — box accuracy and confidence calibration

&#x20; are not the same thing

\- Performance not verified across all species/geographies beyond what's

&#x20; represented in iWildCam's mostly-North-American camera-trap sites



\## Dataset



\[iWildCam 2020](https://www.kaggle.com/c/iwildcam-2020-fgvc7), via the

`qitvision/iwildcam2020-256` Kaggle mirror. \~15,000 labeled images plus mined

hard negatives, evaluated on a held-out validation split.

