#!/usr/bin/env python
import os
import json
import numpy as np
import torch
from transformers import AutoModel
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter

# ---------------------------
# Card utilities
# ---------------------------
RANKS = "23456789TJQKA"

def parse_hand_string(hand_str):
    """Split a string like '2hAc' into ['2h','Ac']"""
    return [hand_str[i:i+2] for i in range(0, len(hand_str), 2)]

# ---------------------------
# Hand evaluator
# ---------------------------
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
# Dataset loader with filtering
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
    # Filter out rare hand ranks
    rank_counts = Counter(y)
    valid_ranks = {rank for rank, count in rank_counts.items() if count >= min_count}
    X_filtered = [x for x, label in zip(X_text, y) if label in valid_ranks]
    y_filtered = np.array([label for label in y if label in valid_ranks])

    return X_filtered, y_filtered

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
    model = AutoModel.from_pretrained("artifacts/checkpoints/run1/best")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Load dataset
    X_text, y = get_player1_data("data", max_samples=300000, min_count=2)
    print(f"Loaded {len(X_text)} samples for probing.")
    if len(X_text) == 0:
        print("No hands found for player1!")
        return

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    # Tokenize
    tokens_train = tokenizer(X_train, return_tensors="pt", padding=True, truncation=True)
    tokens_test  = tokenizer(X_test,  return_tensors="pt", padding=True, truncation=True)

    # Get hidden states
    with torch.no_grad():
        outputs_train = model(**tokens_train, output_hidden_states=True)
        outputs_test  = model(**tokens_test,  output_hidden_states=True)
        hidden_train = outputs_train.hidden_states
        hidden_test  = outputs_test.hidden_states

    # Probe each layer
    for layer_idx, (h_train, h_test) in enumerate(zip(hidden_train, hidden_test)):
        emb_train = h_train.mean(dim=1).cpu().numpy()
        emb_test  = h_test.mean(dim=1).cpu().numpy()
        scaler = StandardScaler()
        emb_train = scaler.fit_transform(emb_train)
        emb_test  = scaler.transform(emb_test)

        clf = LogisticRegression(max_iter=10000)
        clf.fit(emb_train, y_train)
        preds = clf.predict(emb_test)
        acc = accuracy_score(y_test, preds)
        print(f"Layer {layer_idx:02d} accuracy: {acc:.4f}")

        # Confusion matrix
        labels = np.unique(y)
        cm = confusion_matrix(y_test, preds, labels=labels)

        # Plot and save
        os.makedirs("confusion_matrices/handRankBinary", exist_ok=True)
        plt.figure(figsize=(8,6))
        plt.imshow(cm, cmap="Blues", interpolation="nearest")
        plt.title(f"Layer {layer_idx:02d} Confusion Matrix")
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
        plt.savefig(f"confusion_matrices/handRankBinary/confusion_matrix_layer_{layer_idx:02d}.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    main()
