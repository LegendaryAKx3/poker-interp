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
import seaborn as sns

# ---------------------------
# Extract labels from a line
# ---------------------------
def get_label_from_item(item):
    actions = ["cc", "f", "cbr", "sm"]
    for tok in item.strip().split():
        if tok in actions:
            return actions.index(tok)
    return None  # skip if no action found

# ---------------------------
# Remove action token from input
# ---------------------------
def remove_action_token(item):
    actions = ["cc", "f", "cbr", "sm"]
    return " ".join(tok for tok in item.strip().split() if tok not in actions)

# ---------------------------
# Load dataset
# ---------------------------
def load_dataset(data_path, max_samples=None):
    X_text, y = [], []
    for name in os.listdir(data_path):
        if not name.endswith(".ndjson"):
            continue
        with open(os.path.join(data_path, name), "r") as f:
            for line in f:
                try:
                    arr = json.loads(line)
                except:
                    continue
                for item in arr:
                    label = get_label_from_item(item)
                    if label is not None:
                        X_text.append(remove_action_token(item))
                        y.append(label)
                    if max_samples and len(X_text) >= max_samples:
                        return X_text, np.array(y)
    return X_text, np.array(y)

# ---------------------------
# Plot and save confusion matrix
# ---------------------------
def plot_confusion_matrix(cm, layer_idx, save_dir="confusion_matrices"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["cc","f","cbr","sm"],
                yticklabels=["cc","f","cbr","sm"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Layer {layer_idx:02d} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"layer_{layer_idx:02d}_confusion.png"))
    plt.close()

# ---------------------------
# Main
# ---------------------------
def main():
    # Load tokenizer + model
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

    # Load dataset (limit to 10,000 samples for speed)
    X_text, y = load_dataset("data", max_samples=10000)
    print(f"Loaded {len(X_text)} samples for probing.")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    # Tokenize
    tokens_train = tokenizer(X_train, return_tensors="pt", padding=True, truncation=True)
    tokens_test  = tokenizer(X_test,  return_tensors="pt", padding=True, truncation=True)

    # Run model to get hidden states
    with torch.no_grad():
        outputs_train = model(**tokens_train, output_hidden_states=True)
        outputs_test  = model(**tokens_test,  output_hidden_states=True)
        hidden_train = outputs_train.hidden_states
        hidden_test  = outputs_test.hidden_states

    # Probe each layer
    for layer_idx, (layer_train, layer_test) in enumerate(zip(hidden_train, hidden_test)):
        embeddings_train = layer_train.mean(dim=1).cpu().numpy()
        embeddings_test  = layer_test.mean(dim=1).cpu().numpy()

        scaler = StandardScaler()
        embeddings_train = scaler.fit_transform(embeddings_train)
        embeddings_test  = scaler.transform(embeddings_test)

        clf = LogisticRegression(max_iter=10000)
        clf.fit(embeddings_train, y_train)
        preds = clf.predict(embeddings_test)
        acc = accuracy_score(y_test, preds)
        print(f"Layer {layer_idx:02d} accuracy: {acc:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, preds)
        plot_confusion_matrix(cm, layer_idx)

if __name__ == "__main__":
    main()
