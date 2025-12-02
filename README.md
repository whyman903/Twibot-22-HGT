# TwiBot-22 Bot Detection with Heterogeneous Graph Transformers

Multi-modal bot detection combining graph neural networks, transformer text encoders, and profile features on the TwiBot-22 benchmark.

## Overview

Three-branch architecture fusing complementary information sources:

- **Graph**: HGT backbone captures relational patterns across users, tweets, hashtags, and lists
- **Text**: XLM-RoBERTa with LoRA encodes user bios and tweets
- **Profile**: MLP processes 23 engineered user features

## Dataset

TwiBot-22: 1M users, 94M nodes, 340M edges

| Label | Count | Ratio |
|-------|-------|-------|
| Human | 860,057 | 86% |
| Bot | 139,943 | 14% |

## Architecture

```
Heterogeneous Graph ──► HGT (256d, 3 layers) ──┐
User Text ──────────► XLM-RoBERTa (768d) ──────┼──► Concat (1088d) ──► MLP ──► Bot Prob
Profile Features ───► MLP (64d) ──────────────┘
```

**HGT Backbone**: 3 layers, 256 hidden dim, 4 heads, neighbor sampling [20, 10, 5]

**Text Encoder**: `xlm-roberta-base`, LoRA (r=8, alpha=16), 256 max tokens

**Classifier**: 1088 → 512 → 256 → 128 → 2, Focal Loss (gamma=1.5)

## Project Structure

```
src/
├── data/           # Dataset loading and preprocessing
├── models/         # HGT, text encoder, profile MLP, fusion model
├── training/       # Training loop and config
└── utils/          # Losses, metrics, optimizers

scripts/            # SLURM job scripts
train.py            # Training entry point
eval.py             # Evaluation entry point
```

## Usage

```bash
# Train
sbatch scripts/run_train_eval.sh

# Monitor
tensorboard --logdir runs/latest
```

## Citation

```bibtex
@inproceedings{feng2022twibot,
  title={TwiBot-22: Towards Graph-Based Twitter Bot Detection},
  author={Feng, Shangbin and others},
  booktitle={NeurIPS},
  year={2022}
}
```
