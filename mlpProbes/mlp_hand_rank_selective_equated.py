#!/usr/bin/env python
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter
import modal

# ---------------------------
# Modal App Setup
# ---------------------------
app = modal.App("poker-gpt-mlp-probing")
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

    # Flush
    flush_suit = None
    for s, count in suit_counts.items():
        if count >= 5:
            flush_suit = s
            break
    flush_cards = [c for c in all_cards if c[1] == flush_suit] if flush_suit else []

    # Straight
    rank_values = sorted(set([RANKS.index(r) for r in ranks]))
    if 12 in rank_values:
        rank_values.append(-1)  # Ace-low straight
    straight_found = False
    for i in range(len(rank_values)-4):
        if rank_values[i+4] - rank_values[i] == 4:
            straight_found = True
            break

    # Straight flush / Royal flush
    if flush_cards:
        flush_ranks = sorted(set([RANKS.index(c[0]) for c in flush_cards]))
        if 12 in flush_ranks:
            flush_ranks.append(-1)
        for i in range(len(flush_ranks)-4):
            if flush_ranks[i+4] - flush_ranks[i] == 4:
                if flush_ranks[i+4] == 12:
                    return "royal_flush"
                else:
                    return "straight_flush"

    if 4 in rank_counts.values():
        return "four_of_a_kind"
    elif 3 in rank_counts.values() and 2 in rank_counts.values():
        return "full_house"
    elif flush_suit:
        return "flush"
    elif straight_found:
        return "straight"
    elif 3 in rank_counts.values():
        return "three_of_a_kind"
    elif list(rank_counts.values()).count(2) == 2:
        return "two_pair"
    elif 2 in rank_counts.values():
        return "pair"
    else:
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
# MLP Probe
# ---------------------------
class MLPProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ---------------------------
# Modal probing function
# ---------------------------
@app.function(
    image=image,
    gpu="H200",
    cpu=4,
    memory=24*1024,
    timeout=60*60*6,
    volumes={"/data": vol}
)
def run_probing(ckpt_dir: str, tokenizer_dir: str, data_dir: str, max_samples: int = 10000, hidden_dim: int = 256, epochs: int = 10, batch_size: int = 64):
    os.chdir("/data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading tokenizer...")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
    )
    tokenizer.add_special_tokens({
        "pad_token": "<PAD>",
        "unk_token": "<UNK>",
        "bos_token": "<BOS>",
        "eos_token": "<EOS>"
    })

    print("Loading model...")
    model = AutoModel.from_pretrained(ckpt_dir)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)

    print("Loading dataset...")
    X_text, y = get_player1_data(data_dir, max_samples=max_samples, min_count=2)
    print(f"Loaded {len(X_text)} samples.")

    # ---------------------------
    # Balance classes roughly (40th percentile)
    # ---------------------------
    X_text, y = np.array(X_text), np.array(y)
    unique_labels = np.unique(y)
    counts = np.array([np.sum(y == label) for label in unique_labels])
    print("Class counts before balancing:", dict(zip(unique_labels, counts)))

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
    # Train/test split
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.3, random_state=42, stratify=y
    )
    # TODO: test_size was changed to 0.3 from 0.2.

    # Encode labels to integers
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}
    y_train_enc = np.array([label_to_idx[label] for label in y_train])
    y_test_enc  = np.array([label_to_idx[label] for label in y_test])

    # Tokenize inputs
    print("Tokenizing...")
    tokens_train = tokenizer(X_train.tolist(), return_tensors="pt", padding=True, truncation=True)
    tokens_test  = tokenizer(X_test.tolist(),  return_tensors="pt", padding=True, truncation=True)

    # Move to device
    tokens_train = {k: v.to(device) for k, v in tokens_train.items()}
    tokens_test  = {k: v.to(device) for k, v in tokens_test.items()}

    # Extract hidden states
    print("Extracting hidden states...")
    with torch.no_grad():
        outputs_train = model(**tokens_train, output_hidden_states=True)
        outputs_test  = model(**tokens_test,  output_hidden_states=True)
        hidden_train = outputs_train.hidden_states
        hidden_test  = outputs_test.hidden_states

    # ---------------------------
    # Run MLP probe on each layer
    # ---------------------------
    os.makedirs("NeurIPS/confusion_matrices/HandRankSelectEquatedTest30Set", exist_ok=True)

    for layer_idx, (h_train, h_test) in enumerate(zip(hidden_train, hidden_test)):
        emb_train = h_train.mean(dim=1)
        emb_test  = h_test.mean(dim=1)

        scaler = StandardScaler()
        emb_train = torch.tensor(scaler.fit_transform(emb_train.cpu()), dtype=torch.float32).to(device)
        emb_test  = torch.tensor(scaler.transform(emb_test.cpu()), dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train_enc, dtype=torch.long).to(device)
        y_test_t  = torch.tensor(y_test_enc,  dtype=torch.long).to(device)

        probe = MLPProbe(input_dim=emb_train.shape[1], hidden_dim=hidden_dim, num_classes=len(unique_labels)).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(epochs):
            perm = torch.randperm(emb_train.size(0))
            for i in range(0, emb_train.size(0), batch_size):
                idx = perm[i:i+batch_size]
                batch_x = emb_train[idx]
                batch_y = y_train_t[idx]
                optimizer.zero_grad()
                outputs = probe(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluation
        with torch.no_grad():
            logits = probe(emb_test)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y_test_t).float().mean().item()
            print(f"Layer {layer_idx:02d} MLP accuracy: {acc:.4f}")

            cm = confusion_matrix(y_test_enc, preds.cpu(), labels=range(len(unique_labels)))
            plt.figure(figsize=(8,6))
            plt.imshow(cm, cmap="Blues", interpolation="nearest")
            plt.title(f"Layer {layer_idx:02d} Confusion Matrix")
            plt.colorbar()
            tick_marks = np.arange(len(unique_labels))
            plt.xticks(tick_marks, unique_labels, rotation=45)
            plt.yticks(tick_marks, unique_labels)
            plt.ylabel("True")
            plt.xlabel("Predicted")
            for i in range(len(unique_labels)):
                for j in range(len(unique_labels)):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.tight_layout()
            plt.savefig(f"NeurIPS/confusion_matrices/HandRankSelectEquatedTest30Set/confusion_matrix_layer_{layer_idx:02d}.png", dpi=300)
            plt.close()

    print("MLP probing complete. Confusion matrices saved in 'NeurIPS/confusion_matrices/HandRankSelectEquatedTest30Set/'.")

# ---------------------------
# Local entrypoint
# ---------------------------
@app.local_entrypoint()
def main():
    run_probing.remote(
        ckpt_dir="/data/pokerGPT/artifacts/checkpointsNewModel50Epochs/best",
        tokenizer_dir="/data/pokerGPT/artifacts/tokenizer/tokenizer",
        data_dir="/data/pokerGPT/NeurIPS/probeDataTrain",
        max_samples=430000,
        hidden_dim=256,
        epochs=20,
        batch_size=64
    )

if __name__ == "__main__":
    main()
