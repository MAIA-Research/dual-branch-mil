# Joint Three-Class and Dual-Branch TransMIL

This project builds upon the original [TransMIL implementation](https://github.com/szc19990412/transmil). We acknowledge the original authors and extend their work with joint training and attention supervision for whole-slide image classification.

## Overview

The framework jointly trains a three-class model (CCA, HCC and TM) and a dual model with two binary branches:

- **Branch 0:** CCA + TM versus HCC.
- **Branch 1:** CCA versus HCC + TM.

Within the dual model, the branches share a patch-projection layer and CLS token before splitting into independent Transformer pathways. The three-class and dual models receive the same patch features but do not share parameters.

Training combines classification, CCA/HCC reference-attention supervision, TM complementary attention, and alignment between the dual attention union and three-class TM attention. Optional foreground-guided anti-correlation and background suppression are supported.

## Getting Started

Dependencies include PyTorch, PyTorch Lightning, NumPy, pandas, SciPy, scikit-learn, PyYAML, addict, h5py and einops.

Prepare pre-extracted patch features, fold CSVs and attention references, then update their paths and training settings in `config.yaml`. Labels are **0: CCA, 1: HCC, 2: TM**. Use a batch size of 1 for variable-length slide bags.

```bash
# Train one fold.
python train.py --config config.yaml --fold 0

# Train all folds listed in the configuration.
python train.py --config config.yaml
```
For model evaluation and attention heatmap visualization, please refer to the [dual_visualisation repository](https://github.com/MAIA-Research/dual_visualisation).
