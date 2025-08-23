#!/usr/bin/env python
import os
import json
from collections import Counter
from itertools import combinations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Card utilities
# ---------------------------
RANKS = "23456789TJQKA"
SUITS = "cdhs"

def parse_hand_string(hand_str):
    return [hand_str[i:i+2] for i in range(0, len(hand_str), 2)]

def rank_to_value(r):
    return RANKS.index(r)

# ---------------------------
# Hand evaluator
# ---------------------------
def evaluate_hand(hole_cards, board_cards):
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
# Batched embedding extraction
# ---------------------------
def get_embeddings(texts, tokenizer, model, batch_size=64, max_length=128):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        tokens = tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(device)
        with torch.no_grad():
            out = model(**tokens, output_hidden_states=True)
            h = out.last_hidden_state.mean(dim=1)
        embeddings.append(h.cpu().numpy())
    return np.vstack(embeddings)

# ---------------------------
# Main linear probe
# ---------------------------
def main():
    # Tokenizer + model
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_file("artifacts/tokenizer/tokenizer.json")
    )
    tokenizer.add_special_tokens({
        "pad_token": "<PAD>",
        "unk_token": "<UNK>",
        "bos_token": "<BOS>",
        "eos_token": "<EOS>"
    })

    model = AutoModel.from_pretrained("artifacts/checkpoints/run1/best").to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Load data
    X_text, y = get_player1_data("data/hands3.ndjson")
    if len(X_text) == 0:
        raise ValueError("No player1 hands found in dataset!")

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

    # Get embeddings
    emb_train = get_embeddings(X_train.tolist(), tokenizer, model, batch_size=64)
    emb_test  = get_embeddings(X_test.tolist(),  tokenizer, model, batch_size=64)

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
    plt.savefig(f"confusion_matrices/handRank/confusion_matrix.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
