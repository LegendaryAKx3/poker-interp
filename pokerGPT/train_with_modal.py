#!/usr/bin/env python
import os, glob, torch, random, matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup, PreTrainedTokenizerFast
from torch.optim import AdamW
import modal

# ---------------------------
# Modal setup
# ---------------------------
# Create a Modal App
app = modal.App("poker-gpt-training")

# Attach volume for dataset and outputs
vol = modal.Volume.from_name("pokerGPTTSMA")

# Docker image with GPU + dependencies
image = (
    modal.Image.from_registry("pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel")
    .pip_install_from_requirements("requirements.txt")
)

# ---------------------------
# Dataset + Collate
# ---------------------------
class ShardDataset(Dataset):
    def __init__(self, folder):
        self.paths = sorted(glob.glob(os.path.join(folder, "shard_*.pt")))
        self.data = []
        for p in self.paths:
            d = torch.load(p)
            for x, y in zip(d["input_ids"], d["labels"]):
                self.data.append((x, y))
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def collate(batch, pad_id):
    maxlen = max(x[0].size(0) for x in batch)
    bsz = len(batch)
    input_ids = torch.full((bsz, maxlen), pad_id, dtype=torch.long)
    labels    = torch.full((bsz, maxlen), -100, dtype=torch.long)
    attention = torch.zeros((bsz, maxlen), dtype=torch.long)
    for i,(x,y) in enumerate(batch):
        n = x.size(0)
        input_ids[i,:n] = x
        labels[i,:n] = y
        attention[i,:n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention}

def moving_average(x, window=50):
    if len(x) < window:
        return x
    return [sum(x[i-window+1:i+1])/window for i in range(window-1, len(x))]

# ---------------------------
# Modal Training Function
# ---------------------------
@app.function(
    image=image,
    gpu="H200",
    timeout=60*60*12,  # 12 hours
    volumes={"/data": vol},
    cpu=8,
    memory=64*1024
)
def train_model(
    dataset_dir: str,
    tokenizer_dir: str,
    model_name: str = "gpt2",
    out_dir: str = "/data/pokerGPT/checkpoints",
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 5e-5,
    warmup_steps: int = 100,
    save_every: int = 5000,
    grad_accum: int = 2,
    fp16: bool = True,
    val_split: float = 0.05,
    early_stop_patience: int = 3,
    seed: int = 42
):
    # Reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.makedirs(out_dir, exist_ok=True)

    # Tokenizer
    fast_tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(tokenizer_dir, "tokenizer.json"))
    if fast_tok.pad_token is None:
        fast_tok.add_special_tokens({"pad_token": "<PAD>"})
    if fast_tok.eos_token is None:
        fast_tok.add_special_tokens({"eos_token": "<EOS>"})

    # Model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(fast_tok))
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = fast_tok.pad_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Dataset
    ds = ShardDataset(dataset_dir)
    n_val = max(1, int(len(ds) * val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True,
                          collate_fn=lambda b: collate(b, fast_tok.pad_token_id))
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True,
                        collate_fn=lambda b: collate(b, fast_tok.pad_token_id))

    # Optimizer & scheduler
    optim = AdamW(model.parameters(), lr=lr, eps=1e-8, betas=(0.9,0.95))
    num_training_steps = len(train_dl) * epochs // max(1, grad_accum)
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)

    # Tracking
    train_losses_per_step = []
    val_losses_per_epoch = []
    best_val = float("inf")
    stale_epochs = 0
    global_step = 0

    # ---------------------------
    # Training Loop
    # ---------------------------
    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_dl):
            global_step += 1
            batch = {k:v.to(device) for k,v in batch.items()}
            with torch.cuda.amp.autocast(enabled=fp16):
                out = model(**batch)
                loss = out.loss / grad_accum

            scaler.scale(loss).backward()
            update_now = ((step+1) % grad_accum == 0)
            if update_now:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
                sched.step()

            train_losses_per_step.append(float(loss)*grad_accum)

            if save_every and (global_step % save_every == 0):
                step_dir = os.path.join(out_dir, f"step_{global_step}")
                model.save_pretrained(step_dir)
                fast_tok.save_pretrained(step_dir)

        # Validation
        model.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            for batch in val_dl:
                batch = {k:v.to(device) for k,v in batch.items()}
                val_loss_epoch += model(**batch).loss.item()
        val_loss_epoch /= max(1, len(val_dl))
        val_losses_per_epoch.append(val_loss_epoch)
        print(f"Epoch {epoch+1} validation loss: {val_loss_epoch:.4f}")

        if val_loss_epoch < best_val:
            best_val = val_loss_epoch
            stale_epochs = 0
            best_dir = os.path.join(out_dir, "best")
            model.save_pretrained(best_dir)
            fast_tok.save_pretrained(best_dir)
        else:
            stale_epochs += 1
            if stale_epochs >= early_stop_patience:
                print(f"No improvement for {early_stop_patience} epochs. Early stopping.")
                break

# ---------------------------
# Local entrypoint
# ---------------------------
@app.local_entrypoint()
def main():
    train_model.remote(
        dataset_dir="/data/pokerGPT/artifacts/dataset",
        tokenizer_dir="/data/pokerGPT/artifacts/tokenizer",
        out_dir="/data/pokerGPT/artifacts/checkpoints",
        epochs=20,
        batch_size=64
    )

