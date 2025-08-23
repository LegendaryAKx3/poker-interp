#!/usr/bin/env python
import os
import modal
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModel, PreTrainedTokenizerFast
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# ---------------------------
# Modal App
# ---------------------------
app = modal.App("poker-gpt-probing")
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
        "seaborn==0.13.2",
        "modal==1.1.2"
    )
)

# ---------------------------
# Helper functions
# ---------------------------
ACTIONS = ["cc", "f", "cbr", "sm"]

def get_label_from_item(item):
    for tok in item.strip().split():
        if tok in ACTIONS:
            return ACTIONS.index(tok)
    return None

def remove_action_token(item):
    return " ".join(tok for tok in item.strip().split() if tok not in ACTIONS)

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

def plot_confusion_matrix(cm, layer_idx, save_dir="NeurIPS/confusion_matrices/actionIdentification30Test/"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=ACTIONS,
                yticklabels=ACTIONS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Layer {layer_idx:02d} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"layer_{layer_idx:02d}_confusion.png"))
    plt.close()

# ---------------------------
# Modal probing function
# ---------------------------
@app.function(
    image=image,
    gpu="H200",
    cpu=4,
    memory=24 * 1024,
    timeout=60*60*6,
    volumes={"/data": vol}
)
def run_probing(ckpt_dir: str, tokenizer_dir: str, data_dir: str, max_samples: int = 10000):
    os.chdir("/data")
    print("Loading tokenizer...")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(tokenizer_dir, "tokenizer.json")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Loading dataset...")
    X_text, y = load_dataset(data_dir, max_samples)
    print(f"Loaded {len(X_text)} samples.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.3, random_state=42, stratify=y
    )

    print("Tokenizing...")
    tokens_train = tokenizer(X_train, return_tensors="pt", padding=True, truncation=True)
    tokens_test  = tokenizer(X_test,  return_tensors="pt", padding=True, truncation=True)

    # ---- Move tensors to the same device as the model ----
    tokens_train = {k: v.to(device) for k, v in tokens_train.items()}
    tokens_test  = {k: v.to(device) for k, v in tokens_test.items()}

    print("Extracting hidden states...")
    with torch.no_grad():
        outputs_train = model(**tokens_train, output_hidden_states=True)
        outputs_test  = model(**tokens_test,  output_hidden_states=True)
        hidden_train = outputs_train.hidden_states
        hidden_test  = outputs_test.hidden_states

    print("Running probes...")
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

        cm = confusion_matrix(y_test, preds)
        plot_confusion_matrix(cm, layer_idx)

    print("Probing complete. Confusion matrices saved in 'NeurIPS/confusion_matrices/actionIdentification30Test/'.")

# ---------------------------
# Local entrypoint for testing
# ---------------------------
@app.local_entrypoint()
def main():
    run_probing.remote(
        ckpt_dir="/data/pokerGPT/artifacts/checkpointsNewModel50Epochs/best",
        tokenizer_dir="/data/pokerGPT/artifacts/tokenizer/tokenizer/",
        data_dir="/data/pokerGPT/NeurIPS/probeDataTrain",
        max_samples=30000
    )

if __name__ == "__main__":
    main()
