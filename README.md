# Poker Interpretability Research

A comprehensive toolkit for mechanistic interpretability research on poker-playing language models. This repository implements tools for training poker language models, extracting their internal representations, and analyzing how well they learn poker-specific reasoning.

## Overview

This project investigates whether and how language models develop internal representations of poker concepts like hand strength, opponent modeling, and strategic reasoning. The main components include:

- **Poker Hand History (PHH) Generators**: Create realistic 6-max No-Limit Texas Hold'em logs
- **Language Model Training**: Fine-tune GPT-2 style models on poker data with infilling objectives
- **Interpretability Probes**: Linear probes to extract poker concepts from model representations
- **Analysis Tools**: Various utilities for poker equity calculation and hand analysis

## Repository Structure

### Root Level Files

#### Data Generation
- **`gen.py`** - PHH Six-Max Poker Log Generator using CPU-based Monte Carlo equity estimation
- **`gen_torch.py`** - CUDA-accelerated version with PyTorch for faster equity calculations and batch processing

### pokerGPT/ Directory

#### Core Training Pipeline
- **`poker_tokenizer.py`** - Build custom domain tokenizer with poker-specific vocabulary (cards, actions, bet sizes)
- **`preprocess.py`** - Convert NDJSON poker logs to tokenized training sequences with infilling format
- **`train.py`** - Fine-tune GPT-2 style models on preprocessed poker data
- **`infer.py`** - Fill-in-the-blank inference for trained poker models

#### Linear Probes & Analysis
- **`linear_probe_equity.py`** - Train linear regression probes to predict Monte Carlo equity from model representations
- **`linear_probe_for_action.py`** - Probe for action prediction (fold/call/bet) from internal states  
- **`linear_probe_for_hand_rank.py`** - Extract hand strength representations
- **`linear_probe_for_hand_rank_select.py`** - Selective hand rank probing with advanced techniques
- **`mlp_probe_equity.py`** - Multi-layer perceptron probes for equity prediction
##### Note: Some probes require the use of Modal, here is the [Modal Documentation](https://modal.com/docs)

#### Utilities
- **`formatFromNDJSON.py`** - Converts a single playthrough into the proper formatting for the model inference.
- **`preprocess_equity_dataset.py`** - Preprocessing pipeline for equity prediction tasks

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
cd pokerGPT
pip install -r requirements.txt
```

### 2. Generate Training Data

```bash
# Generate poker hand histories (CPU version)
python gen.py --hands 1000 --seed 42 --out data/generated_hands.ndjson

# Or use GPU-accelerated version for larger datasets (This will be nessecary for the required dataset scale)
python gen_torch.py --hands 5000 --device cuda --out data/large_dataset.ndjson
```

### 3. Train a Poker Language Model

```bash
cd pokerGPT

# Build custom tokenizer
python poker_tokenizer.py --data data --out artifacts/tokenizer

# Preprocess data with infilling format
python preprocess.py \
    --data data \
    --tokenizer artifacts/tokenizer \
    --out artifacts/dataset \
    --max-seq 1024 \
    --mask-prob 0.5

# Train the model
python train.py \
    --dataset artifacts/dataset \
    --tokenizer artifacts/tokenizer \
    --model gpt2 \
    --out artifacts/checkpoints/poker_model \
    --epochs 3 \
    --batch 32 \
    --lr 5e-5
```

### 4. Train Interpretability Probes

```bash
# Train equity prediction probe
cd pokerGPT
python mlp_probe_equity.py \
    --model artifacts/checkpoints/poker_model \
    --data data/hands.ndjson \
    --output_dir probe_results/equity
 # If using Modal, use modal run <script_name>
 # Note that you may have to change some of the Modal code (such as name) to adapt it to your Modal volume.
```

### 5. Run Inference

```bash
cd pokerGPT

# Fill-in-the-blank poker scenarios
python infer.py \
    --ckpt artifacts/checkpoints/poker_model \
    --tokenizer artifacts/tokenizer \
    --context "d dh p1 AhAc | <GAP> | d db 7h8c9s" \
    --max-new-tokens 50
```

## Data Formats

### Poker Hand History (PHH)
```json
["d dh p1 AhAc", "d dh p2 Kh7c", "p1 cbr 300", "p2 cc", "d db 9s8c2h", "p1 cbr 600", "p2 f"]
```

### Infilling Format (Training)
```
Original: "p1 cc | p2 cbr 500 | p3 f"
Training: "p1 cc | <GAP> | p3 f <ANS> p2 cbr 500 <EOS>"
```






