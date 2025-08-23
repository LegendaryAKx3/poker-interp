#!/usr/bin/env python
import os
import modal
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModel, PreTrainedTokenizerFast
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

def load_dataset(data_path, max_samples=None, balance=True, percentile=40):
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
                        X_text, y = np.array(X_text), np.array(y)
                        return balance_dataset(X_text, y, percentile) if balance else (X_text, y)
    X_text, y = np.array(X_text), np.array(y)
    return balance_dataset(X_text, y, percentile) if balance else (X_text, y)

def balance_dataset(X_text, y, percentile=40):
    from collections import Counter
    rng = np.random.default_rng(42)

    counts = Counter(y)
    print("Original class distribution:", counts)

    target_count = max(int(np.percentile(list(counts.values()), percentile)), 10)
    print(f"Target per-class count: {target_count}")

    X_balanced, y_balanced = [], []
    for action in np.unique(y):
        idxs = np.where(y == action)[0]
        if len(idxs) > target_count:
            idxs = rng.choice(idxs, size=target_count, replace=False)
        for i in idxs:
            X_balanced.append(X_text[i])
            y_balanced.append(y[i])

    idxs = rng.permutation(len(X_balanced))
    X_balanced = np.array(X_balanced)[idxs]
    y_balanced = np.array(y_balanced)[idxs]

    print("Balanced class distribution:", Counter(y_balanced))
    return X_balanced.tolist(), y_balanced

def plot_confusion_matrix(cm, layer_idx, save_dir="NeurIPS/confusion_matrices/actionIdentificationEquated30TestMLP150k/"):
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
# One-hidden-layer MLP Probe
# ---------------------------
class MLPProbe(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=4):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def train_mlp_probe(X_train, y_train, X_val, y_val, num_epochs=15, lr=1e-3, batch_size=256, device="cpu"):
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.long, device=device)
    X_val   = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val   = torch.tensor(y_val, dtype=torch.long, device=device)

    model = MLPProbe(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # Validation accuracy
        model.eval()
        with torch.no_grad():
            preds = model(X_val).argmax(dim=1)
            acc = (preds == y_val).float().mean().item()
        print(f"Epoch {epoch+1}/{num_epochs} - loss: {total_loss:.4f} - val acc: {acc:.4f}")

    return model

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
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Tokenizing...")
    tokens_train = tokenizer(X_train, return_tensors="pt", padding=True, truncation=True)
    tokens_test  = tokenizer(X_test,  return_tensors="pt", padding=True, truncation=True)

    tokens_train = {k: v.to(device) for k, v in tokens_train.items()}
    tokens_test  = {k: v.to(device) for k, v in tokens_test.items()}

    print("Extracting hidden states...")
    with torch.no_grad():
        outputs_train = model(**tokens_train, output_hidden_states=True)
        outputs_test  = model(**tokens_test,  output_hidden_states=True)
        hidden_train = outputs_train.hidden_states
        hidden_test  = outputs_test.hidden_states

    print("Running one-hidden-layer MLP probes...")
    for layer_idx, (layer_train, layer_test) in enumerate(zip(hidden_train, hidden_test)):
        embeddings_train = layer_train.mean(dim=1).cpu().numpy()
        embeddings_test  = layer_test.mean(dim=1).cpu().numpy()

        scaler = StandardScaler()
        embeddings_train = scaler.fit_transform(embeddings_train)
        embeddings_test  = scaler.transform(embeddings_test)

        mlp_model = train_mlp_probe(
            embeddings_train, y_train,
            embeddings_test, y_test,
            num_epochs=15, lr=1e-3, device=device
        )

        # final evaluation
        mlp_model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(embeddings_test, dtype=torch.float32, device=device)
            preds = mlp_model(X_test_tensor).argmax(dim=1).cpu().numpy()

        acc = accuracy_score(y_test, preds)
        print(f"Layer {layer_idx:02d} accuracy: {acc:.4f}")

        cm = confusion_matrix(y_test, preds)
        plot_confusion_matrix(cm, layer_idx)

    print("Probing complete. Confusion matrices saved in 'NeurIPS/confusion_matrices/actionIdentificationEquated30TestMLP150k/'.")

# ---------------------------
# Local entrypoint for testing
# ---------------------------
@app.local_entrypoint()
def main():
    run_probing.remote(
        ckpt_dir="/data/pokerGPT/artifacts/checkpointsNewModel50Epochs/best",
        tokenizer_dir="/data/pokerGPT/artifacts/tokenizer/tokenizer/",
        data_dir="/data/pokerGPT/NeurIPS/probeDataTrain",
        max_samples=150000
    )

if __name__ == "__main__":
    main()
