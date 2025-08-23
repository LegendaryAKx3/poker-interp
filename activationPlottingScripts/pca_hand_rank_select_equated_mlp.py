#!/usr/bin/env python
import os
import json
import numpy as np
import torch
from transformers import AutoModel, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from collections import Counter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import modal

# ---------------------------
# Modal App Setup
# ---------------------------
app = modal.App("pokerGPT-activation-pca")
vol = modal.Volume.from_name("pokerGPTTSMA")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel")
    .pip_install(
        "transformers>=4.41.0",
        "tokenizers>=0.15.2",
        "torch>=2.2.0",
        "numpy>=1.24.0",
        "matplotlib==3.10.3",
        "scikit-learn==1.7.0",
        "modal==1.1.2"
    )
)

# ---------------------------
# Card utilities
# ---------------------------
RANKS = "23456789TJQKA"

def parse_hand_string(hand_str):
    return [hand_str[i:i+2] for i in range(0, len(hand_str), 2)]

def evaluate_hand(hole_cards, board_cards):
    all_cards = hole_cards + board_cards
    ranks = [c[0] for c in all_cards]
    suits = [c[1] for c in all_cards]
    rank_counts = {r: ranks.count(r) for r in set(ranks)}
    suit_counts = {s: suits.count(s) for s in set(suits)}

    flush_suit = next((s for s, count in suit_counts.items() if count >= 5), None)
    flush_cards = [c for c in all_cards if c[1] == flush_suit] if flush_suit else []

    rank_values = sorted(set([RANKS.index(r) for r in ranks]))
    if 12 in rank_values:
        rank_values.append(-1)
    straight_found = any(rank_values[i+4] - rank_values[i] == 4 for i in range(len(rank_values)-4))

    if flush_cards:
        flush_ranks = sorted(set([RANKS.index(c[0]) for c in flush_cards]))
        if 12 in flush_ranks:
            flush_ranks.append(-1)
        for i in range(len(flush_ranks)-4):
            if flush_ranks[i+4] - flush_ranks[i] == 4:
                return "royal_flush" if flush_ranks[i+4] == 12 else "straight_flush"

    if 4 in rank_counts.values(): return "four_of_a_kind"
    if 3 in rank_counts.values() and 2 in rank_counts.values(): return "full_house"
    if flush_suit: return "flush"
    if straight_found: return "straight"
    if 3 in rank_counts.values(): return "three_of_a_kind"
    if list(rank_counts.values()).count(2) == 2: return "two_pair"
    if 2 in rank_counts.values(): return "pair"
    return "high_card"

# ---------------------------
# Dataset loader
# ---------------------------
def get_player1_data(data_path, max_samples=None, min_count=2):
    X_text, y = [], []
    for fname in os.listdir(data_path):
        if not fname.endswith(".ndjson"):
            continue
        with open(os.path.join(data_path, fname), "r") as f:
            for line in f:
                try:
                    arr = json.loads(line)
                except:
                    continue
                hole_cards, board_cards = [], []
                for item in arr:
                    if item.startswith("d dh p1 "):
                        hole_cards = parse_hand_string(item.split()[-1])
                    elif item.startswith("d db "):
                        board_cards = parse_hand_string(item.split()[-1])
                if hole_cards and board_cards:
                    rank = evaluate_hand(hole_cards, board_cards)
                    X_text.append(" ".join(hole_cards + board_cards))
                    y.append(rank)
                    if max_samples and len(X_text) >= max_samples:
                        break
    y = np.array(y)
    rank_counts = Counter(y)
    valid_ranks = {rank for rank, count in rank_counts.items() if count >= min_count}
    X_filtered = [x for x, label in zip(X_text, y) if label in valid_ranks]
    y_filtered = np.array([label for label in y if label in valid_ranks])
    return X_filtered, y_filtered

# ---------------------------
# Modal PCA function
# ---------------------------
@app.function(
    image=image,
    gpu="H200",
    cpu=4,
    memory=24*1024,
    timeout=60*60*6,
    volumes={"/data": vol}
)
def run_pca_visualization(ckpt_dir: str, tokenizer_dir: str, data_dir: str, max_samples: int = 10000):
    os.chdir("/data")
    print("Loading tokenizer...")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
    )
    tokenizer.add_special_tokens({"pad_token":"<PAD>", "unk_token":"<UNK>", "bos_token":"<BOS>", "eos_token":"<EOS>"})

    print("Loading model...")
    model = AutoModel.from_pretrained(ckpt_dir)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Loading dataset...")
    X_text, y = get_player1_data(data_dir, max_samples=max_samples, min_count=2)
    print(f"Loaded {len(X_text)} samples.")

    # ---------------------------
    # Balance dataset by 40th percentile
    # ---------------------------
    X_text, y = np.array(X_text), np.array(y)
    unique_labels = np.unique(y)
    counts = np.array([np.sum(y == label) for label in unique_labels])
    target_count = max(int(np.percentile(counts, 40)), 10)
    print("Target samples per class (approx.):", target_count)

    balanced_X, balanced_y = [], []
    for label in unique_labels:
        idx = np.where(y == label)[0]
        if len(idx) > target_count:
            idx = np.random.choice(idx, target_count, replace=False)
        balanced_X.extend(X_text[idx])
        balanced_y.extend(y[idx])

    X_text, y = np.array(balanced_X), np.array(balanced_y)
    counts_after = {label: np.sum(y == label) for label in unique_labels}
    print("Class counts after balancing:", counts_after)

    # ---------------------------
    # Tokenize
    # ---------------------------
    tokens = tokenizer(list(X_text), return_tensors="pt", padding=True, truncation=True)
    tokens = {k: v.to(device) for k,v in tokens.items()}

    # ---------------------------
    # Extract hidden states
    # ---------------------------
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple[layer][batch, seq_len, hidden_dim]

    # ---------------------------
    # PCA + Scatterplot
    # ---------------------------
    os.makedirs("pca_plots/hand_rank_select_equated_mlp200kTest", exist_ok=True)
    for layer_idx, hidden_layer in enumerate(hidden_states):
        hidden_avg = hidden_layer.mean(dim=1)  # [batch, hidden_dim]
        pca = PCA(n_components=2)
        hidden_2d = pca.fit_transform(hidden_avg.cpu().numpy())

        plt.figure(figsize=(8,6))
        unique_ranks = np.unique(y)
        cmap = plt.cm.get_cmap("tab10", len(unique_ranks))
        for i, rank in enumerate(unique_ranks):
            idx = np.array(y) == rank
            plt.scatter(hidden_2d[idx, 0], hidden_2d[idx, 1], label=rank, alpha=0.7, color=cmap(i))
        plt.title(f"PCA of Layer {layer_idx} Activations")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"pca_plots/hand_rank_select_equated_mlp200kTest/pca_layer_{layer_idx:02d}.png", dpi=300)
        plt.close()

    print("PCA plots saved in 'pca_plots/hand_rank_select_equated_mlp200kTest'")

# ---------------------------
# Local entrypoint
# ---------------------------
@app.local_entrypoint()
def main():
    run_pca_visualization.remote(
        ckpt_dir="/data/pokerGPT/artifacts/checkpoints/run1/best",
        tokenizer_dir="/data/pokerGPT/artifacts/tokenizer",
        data_dir="/data/pokerGPT/probeData",
        max_samples=200000
    )

if __name__ == "__main__":
    main()
