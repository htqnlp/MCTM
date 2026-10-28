# MCTM

Research code accompanying **Event Representation Learning with Multi-Grained Contrastive Learning and Triple-Mixture of Experts**.

Tianqi Hu, Lishuang Li, Xueyang Qin, and Yubo Feng. **LREC-COLING 2024**, pages 6643-6654.

[Paper and citation](https://aclanthology.org/2024.lrec-main.588/)

## Overview

MCTM learns event representations using three masking granularities: random, subject-predicate, and predicate-object. A mixture-of-experts component learns the importance of different granularities for contrastive learning.

## Repository contents

- `main.py`: training and evaluation entry point.
- `model_moe.py`: mixture-of-experts model and sparse gating.
- `model.py`: event representation and memory-bank components.
- `span_utils.py`, `data_utils.py`: span and dataset processing.
- `config_model.py`, `config_data.py`: model and dataset configuration.
- `eva_mcnc.py`, `eval_mcnc2.py`: MCNC evaluation scripts.
- `data_utils/event_eval/`: event-similarity evaluation files.
- `mcnc_data/`: selected evaluation files.

## Environment and configuration

This repository contains a research-code snapshot. The original scripts and dependency list are retained. The inherited setup used Python 3.8 and PyTorch with Texar-PyTorch.

Review `requirements.txt` before installation: it contains unpinned dependencies and legacy package names. Configure the data paths and vocabulary in `config_data.py`, and check model settings against the paper before attempting reproduction. The current local defaults are development settings and are not a claim of exact reproduction settings.

## Data and checkpoints

The large training/validation JSON files, raw Ollie extraction corpora, model checkpoints, logs, and local editor/cache files are excluded from Git.

Place the training and validation files at the paths configured in `config_data.py`, including:

```text
data_utils/multi_pos_train3.json
data_utils/valid.json
```

The inherited dataset instructions are preserved in [the original SWCC README](docs/SWCC_README.md). Follow the dataset providers' access and usage requirements. No trained checkpoint is included in this repository.

## Entry points

After preparing compatible dependencies, data, and configuration, the supplied entry points are:

```bash
# Training
CUDA_VISIBLE_DEVICES=0 python main.py --do-train

# Evaluation; replace the path with your trained checkpoint
CUDA_VISIBLE_DEVICES=0 python main.py --do-eval --checkpoint ./models/checkpoint.pt

# MCNC evaluation; follows test.sh
CUDA_VISIBLE_DEVICES=0 python eva_mcnc.py --do-eval --checkpoint ./models/checkpoint.pt
```

These commands document the supplied scripts; a full training or reproduction run has not been verified for this upload.

## Acknowledgements

This codebase builds on **SWCC: Improving Event Representation via Simultaneous Weakly Supervised Contrastive Learning and Clustering** by Jun Gao, Wei Wang, Changlong Yu, Huan Zhao, Wilfred Ng, and Ruifeng Xu (ACL 2022). Its original README, data instructions, and citation are retained in [docs/SWCC_README.md](docs/SWCC_README.md).

Please retain upstream attribution and follow the applicable source-code and dataset terms when reusing the materials.

## Citation

```bibtex
@inproceedings{hu-etal-2024-event-representation,
  title = {Event Representation Learning with Multi-Grained Contrastive Learning and Triple-Mixture of Experts},
  author = {Hu, Tianqi and Li, Lishuang and Qin, Xueyang and Feng, Yubo},
  booktitle = {Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
  year = {2024},
  pages = {6643--6654},
  publisher = {ELRA and ICCL},
  url = {https://aclanthology.org/2024.lrec-main.588/}
}
```
