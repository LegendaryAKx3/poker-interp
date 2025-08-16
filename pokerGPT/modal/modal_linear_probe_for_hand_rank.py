#!/usr/bin/env python
import os
import json
import numpy as np
import torch
from transformers import AutoModel, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations
import modal

# ---------------------------
# Modal App Setup
# ---------------------------
app = modal.App("poker-gpt-hand-probing")
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
SUITS = "cdhs"

def parse_hand_string(hand_str):
    return [hand_str[i:i+2] for i in range(0, len(hand_str), 2)]

def rank_to_value(r):
    return RANKS.index(r)

def evaluate_hand(hole_cards, board_cards):
    from collections import Counter
    from itertools import combinations

    all_cards = hole_cards + board_cards
    best_rank = "high_card"
    hand_order = [
        "high_card", "pair", "two_pair", "three_of_a_kind",
        "straight", "flush", "full_house", "four_of_a_kind",
        "straight_flush", "royal_flush"
    ]

    def classify_5cards(cards):
        ranks = [c[0] for c in cards]
        suits = [c[1] for c in cards]
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)

        is_flush = max(suit_counts.values()) == 5
        rank_values = sorted([rank_to_value(r) for r in ranks], reverse=True)

        # Ace-low straight
        if set([12, 0, 1, 2, 3]).issubset(rank_values):
            is_straight = True
            high_card = 3
        else:
            is_straight = all(rank_values[i] - rank_values[i+1] == 1 for i in range(4))
            high_card = rank_values[0]

        if is_flush and is_straight:
            return "royal_flush" if high_card == 12 else "straight_flush"
        if 4 in rank_counts.values():
            return "four_of_a_kind"
        if 3 in rank_counts.values() and 2 in rank_counts.values():
            return "full_house"
        if is_flush:
            return "flush"
        if is_straight:
            return "straight"
        if 3 in rank_counts.values():
            return "three_of_a_kind"
        if list(rank_counts.values()).count(2) == 2:
            return "two_pair"
        if 2 in rank_counts.values():
            return "pair"
        return "high_card"

    for comb in combinations(all_cards, 5):
        rank = classify_5cards(list(comb))
        if hand_order.index(rank) > hand_order.index(best_rank):
            best_rank = rank
            if best_rank == "royal_flush":
                break
    return best_rank

# ---------------------------
# Load hands from NDJSON
# ---------------------------
def get_player1_data(file_path, max_samples=None):
    X_text, y = [], []
    hand_counter = Counter()
    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                arr = json.loads(line)
            except json.JSONDecodeError:
                continue

            hole_cards, board_cards = [], []
            for item in arr:
                if item.startswith("d dh p1 "):
                    hole_cards = parse_hand_string(item.split()[-1])
                elif item.startswith("d db "):
                    board_cards.extend(parse_hand_string(item.split()[-1]))

            if hole_cards and board_cards:
                rank = evaluate_hand(hole_cards, board_cards)
                X_text.append(" ".join(hole_cards + board_cards))
                y.append(rank)
                hand_counter[rank] += 1
                if max_samples and len(X_text) >= max_samples:
                    break

    print("Hand distribution:", hand_counter)
    return X_text, np.array(y)

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
def run_probing(ckpt_dir: str, tokenizer_dir: str, data_file: str, max_samples: int = 300000):
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
    X_text, y = get_player1_data(data_file, max_samples=max_samples)
    print(f"Loaded {len(X_text)} samples.")

    # Filter classes with <2 samples
    labels, counts = np.unique(y, return_counts=True)
    valid_labels = labels[counts >= 2]
    mask = np.isin(y, valid_labels)
    X_text, y = np.array(X_text)[mask], np.array(y)[mask]

    if len(np.unique(y)) < 2:
        raise ValueError("Not enough distinct hand categories for classification.")

    print("Classes:", np.unique(y))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    # Tokenize and embed
    def get_embeddings(texts):
        embeddings = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            tokens = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                out = model(**tokens, output_hidden_states=True)
                h = out.last_hidden_state.mean(dim=1)
            embeddings.append(h.cpu().numpy())
        return np.vstack(embeddings)

    emb_train = get_embeddings(X_train.tolist())
    emb_test  = get_embeddings(X_test.tolist())

    # Standardize
    scaler = StandardScaler()
    emb_train = scaler.fit_transform(emb_train)
    emb_test  = scaler.transform(emb_test)

    # Logistic regression probe
    clf = LogisticRegression(max_iter=100000)
    clf.fit(emb_train, y_train)
    preds = clf.predict(emb_test)
    acc = accuracy_score(y_test, preds)
    print(f"Probe accuracy: {acc:.4f}")

    # Confusion matrix
    labels = np.unique(y)
    cm = confusion_matrix(y_test, preds, labels=labels)

    os.makedirs("confusion_matrices/handRank", exist_ok=True)
    plt.figure(figsize=(10,8))
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.title(f"Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig("confusion_matrices/handRank/confusion_matrix.png", dpi=300)
    plt.close()
    print("Confusion matrix saved to 'confusion_matrices/handRank/'.")

# ---------------------------
# Local entrypoint
# ---------------------------
@app.local_entrypoint()
def main():
    run_probing.remote(
        ckpt_dir="/data/pokerGPT/artifacts/checkpoints/run1/best",
        tokenizer_dir="/data/pokerGPT/artifacts/tokenizer",
        data_file="/data/pokerGPT/probeData/hands.ndjson",
        max_samples=30000
    )

if __name__ == "__main__":
    main()
