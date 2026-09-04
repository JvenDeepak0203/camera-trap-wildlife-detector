\# Camera Trap Wildlife Detector



A lightweight image classifier that detects whether a camera-trap photo contains

an animal or is an empty/false-triggered frame. Built to address a real bottleneck

in wildlife conservation: camera traps generate huge volumes of images, most of

which are empty, and manually reviewing them all is a major time sink for researchers.



\## How it works



A ResNet18 (pretrained on ImageNet) fine-tuned on a labeled subset of the

\[iWildCam 2020](https://www.kaggle.com/c/iwildcam-2020-fgvc7) camera trap dataset,

using per-image detector annotations to build binary labels (animal vs. empty).



\## Results



Evaluated on a held-out validation set of 3,000 labeled images never seen during training:



| Metric              | Empty | Animal |

|---------------------|-------|--------|

| Precision           | 0.75  | 0.88   |

| Recall              | 0.84  | 0.81   |

| F1-score            | 0.79  | 0.85   |



Overall accuracy: 82%, compared to a 60% baseline from always guessing "animal"

(the majority class).



\*\*Note:\*\* an earlier version trained on a smaller (3,000-image) sample achieved

higher recall on the "animal" class (90% vs. 81% here), which matters more for

this tool's use case since missing an actual animal frame is a costlier error

than a false alarm on an empty frame. This suggests dataset size alone doesn't

guarantee better real-world performance, and further tuning (more epochs, class

weighting) could likely recover that recall while keeping the larger dataset's gains.



\## Running it locally



```bash

python -m venv venv

venv\\Scripts\\activate      # Windows

pip install -r requirements.txt

python app.py

```



Then open the printed local URL in your browser.



\## Limitations



Trained on a sample of one dataset from specific camera trap locations — accuracy

on very different environments, lighting conditions, or camera setups may vary.

Not tested on video, only still images.

