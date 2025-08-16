#!/usr/bin/env python
import argparse, os, glob, torch, random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

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

# ---------------------------
# Moving Average
# ---------------------------
def moving_average(x, window=50):
    if len(x) < window:
        return x
    return [sum(x[i-window+1:i+1])/window for i in range(window-1, len(x))]

# ---------------------------
# Training Function
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--val-split", type=float, default=0.05)
    ap.add_argument("--early-stop-patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Tokenizer
    from transformers import PreTrainedTokenizerFast
    fast_tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(args.tokenizer, "tokenizer.json"))
    if fast_tok.pad_token is None:
        fast_tok.add_special_tokens({"pad_token": "<PAD>"})
    if fast_tok.eos_token is None:
        fast_tok.add_special_tokens({"eos_token": "<EOS>"})

    # Model
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.resize_token_embeddings(len(fast_tok))
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = fast_tok.pad_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Dataset
    ds = ShardDataset(args.dataset)
    n_val = max(1, int(len(ds) * args.val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=8, pin_memory=True,
                          collate_fn=lambda b: collate(b, fast_tok.pad_token_id))
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=4, pin_memory=True,
                        collate_fn=lambda b: collate(b, fast_tok.pad_token_id))

    # Optimizer & scheduler
    optim = AdamW(model.parameters(), lr=args.lr, eps=1e-8, betas=(0.9,0.95))
    num_training_steps = len(train_dl) * args.epochs // max(1, args.grad_accum)
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=args.warmup, num_training_steps=num_training_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # Tracking
    train_losses_per_step = []
    val_losses_per_epoch = []
    best_val = float("inf")
    stale_epochs = 0
    global_step = 0

    # Setup live plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(12,6))
    line_train, = ax.plot([], [], label='Train Loss (smoothed)')
    line_val, = ax.plot([], [], 'rx', label='Validation Loss')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.grid(True)
    ax.legend()

    # ---------------------------
    # Training Loop
    # ---------------------------
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}")
        optim.zero_grad()
        for step, batch in enumerate(pbar):
            global_step += 1
            batch = {k:v.to(device) for k,v in batch.items()}

            with torch.cuda.amp.autocast(enabled=args.fp16):
                out = model(**batch)
                loss = out.loss / args.grad_accum

            scaler.scale(loss).backward()
            update_now = ((step+1) % args.grad_accum == 0)
            if update_now:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
                sched.step()

            train_losses_per_step.append(float(loss)*args.grad_accum)
            smoothed = moving_average(train_losses_per_step, window=50)

            # Update live plot
            line_train.set_data(range(len(smoothed)), smoothed)
            line_val.set_data([len(train_dl)*(i+1) for i in range(len(val_losses_per_epoch))], val_losses_per_epoch)
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.001)

            pbar.set_postfix({"loss": f"{train_losses_per_step[-1]:.4f}"})
            if args.save_every and (global_step % args.save_every == 0):
                step_dir = os.path.join(args.out, f"step_{global_step}")
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
            best_dir = os.path.join(args.out, "best")
            model.save_pretrained(best_dir)
            fast_tok.save_pretrained(best_dir)
        else:
            stale_epochs += 1
            if stale_epochs >= args.early_stop_patience:
                print(f"No improvement for {args.early_stop_patience} epochs. Early stopping.")
                break

        # Save epoch
        epoch_dir = os.path.join(args.out, f"epoch_{epoch+1}")
        model.save_pretrained(epoch_dir)
        fast_tok.save_pretrained(epoch_dir)

    # Final save + static plot
    _plot_final(train_losses_per_step, val_losses_per_epoch, len(train_dl), args.out)
    model.save_pretrained(args.out)
    fast_tok.save_pretrained(args.out)
    print("Training complete. Saved final model to", args.out)
    plt.ioff()
    plt.show()

# ---------------------------
# Final static plot
# ---------------------------
def _plot_final(train_losses, val_losses, steps_per_epoch, outdir, filename="loss_plot_final.png"):
    plt.figure(figsize=(12,6))
    smoothed = moving_average(train_losses, window=50)
    plt.plot(range(len(smoothed)), smoothed, label='Train Loss (smoothed)')
    if val_losses and steps_per_epoch > 0:
        val_steps = [steps_per_epoch*(i+1) for i in range(len(val_losses))]
        plt.plot(val_steps, val_losses, 'rx', label='Validation Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Final Training & Validation Loss')
    plt.grid(True)
    plt.legend()
    os.makedirs(outdir, exist_ok=True)
    plt.tight_layout()
    path = os.path.join(outdir, filename)
    plt.savefig(path, dpi=150)
    print("Saved final loss plot at", path)

if __name__ == "__main__":
    main()
