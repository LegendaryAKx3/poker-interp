
# usage: python poker-gpt.py --data_dir /path/to/phh_texts --save_dir ./ckpt 
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import string
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import ast

# ────────────────────────────────────────────────────────────────────────────────
# 1.  Character-level tokenizer
# ────────────────────────────────────────────────────────────────────────────────
class CharTokenizer:
    """Simple stateless char-to-int and int-to-char mapper."""

    def __init__(self, extra_chars: str = "") -> None:
        base = (
            string.ascii_letters
            + string.digits
            + string.punctuation
            + " \t\n"  # space, tab, newline
        )
        charset = sorted(set(base + extra_chars))
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(charset)}
        self.itos: List[str] = charset

    def encode(self, text: str) -> List[int]:  # noqa: D401
        print("text: ", text)
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids: List[int]) -> str:  # noqa: D401
        return "".join(self.itos[i] for i in ids)

    @property
    def vocab_size(self) -> int:  # noqa: D401
        return len(self.itos)


# ────────────────────────────────────────────────────────────────────────────────
# 2.  PHH parsing utilities – simplified key-value extraction
# ────────────────────────────────────────────────────────────────────────────────

_FIELD_PAT = re.compile(r"^(\w+)\s*=\s*(.+)$")


def _parse_list(value: str):
    """Safely parse a Python-literal list from the PHH line value."""
    try:
        return ast.literal_eval(value)
    except Exception:
        return []


def extract_relevant_lines(filepath: Path) -> str:
    """Return cleaned text consisting of numeric context + action strings.

    Extracts:
        blinds_or_straddles, starting_stacks, winnings, antes (if present)
        actions – list of action strings
    Concatenates them into a single newline-separated chunk followed by <END_HAND>.
    """
    blinds = stacks = winnings = antes = []
    actions: List[str] = []

    with filepath.open() as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw or raw.startswith("["):
                # skip index headers like "[1]" or blank lines
                continue
            m = _FIELD_PAT.match(raw)
            if not m:
                continue
            key, val = m.group(1), m.group(2)
            if key == "blinds_or_straddles":
                blinds = _parse_list(val)
            elif key == "starting_stacks":
                stacks = _parse_list(val)
            elif key == "winnings":
                winnings = _parse_list(val)
            elif key == "antes":
                antes = _parse_list(val)
            elif key == "actions":
                actions = _parse_list(val)

    # Build textual representation: one line each for context, then actions lines
    parts = []
    if blinds:
        parts.append(f"BLINDS {blinds}")
    if antes:
        parts.append(f"ANTES {antes}")
    if stacks:
        parts.append(f"STACKS {stacks}")
    if winnings:
        parts.append(f"WINNINGS {winnings}")

    parts.extend(actions)  # each action string becomes its own line

    return "\n".join(str(p) for p in parts) + "\n<END_HAND>\n"


class PokerPHHDataset(Dataset):
    """Yield training examples sampled *within* a single hand history.

    Each `__getitem__` picks **one** hand (deterministically by index) then
    crops a random contiguous span of length `block_size+1` from it.  This
    prevents the model from attending across `<END_HAND>` boundaries.
    """

    def __init__(self, dir_path: str | Path, tokenizer: CharTokenizer, block_size: int = 256):
        self.block_size = block_size
        self.tokenizer = tokenizer

        dir_path = Path(dir_path)
        assert dir_path.is_dir(), f"{dir_path} is not a directory"

        hand_texts: List[str] = []
        print("Scanning PHH files …")
        for fp in dir_path.glob("**/*"):
            if fp.suffix.lower() in {".phh", ".txt"}:
                hand_texts.append(extract_relevant_lines(fp))

        if not hand_texts:
            raise ValueError(f"No .phh/.txt files found in {dir_path}")

        # encode each hand separately
        self.hands: List[torch.Tensor] = []
        for txt in hand_texts:
            ids = tokenizer.encode(txt)
            if len(ids) >= block_size + 1:  # keep only sufficiently long hands
                self.hands.append(torch.tensor(ids, dtype=torch.long))

        print(f"Loaded {len(self.hands)} hands (>= {block_size+1} chars)")

    def __len__(self):
        return len(self.hands)

    def __getitem__(self, idx):
        hand = self.hands[idx]
        if len(hand) == self.block_size + 1:
            chunk = hand
        else:
            start = random.randint(0, len(hand) - self.block_size - 1)
            chunk = hand[start : start + self.block_size + 1]
        return chunk[:-1], chunk[1:]  # (input, target)


class GPTConfig:
    def __init__(self, vocab_size: int, n_layer=4, n_head=4, n_embd=256, block_size=256):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size

# lets see if this works, else we can scale up the architecture too 

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()

        print("vocab size: cfg.vocab_size", cfg.vocab_size)
        print("n_embd: cfg.n_embd", cfg.n_embd)
        print("block_size: cfg.block_size", cfg.block_size)
        print("n_layer: cfg.n_layer", cfg.n_layer)
        print("n_head: cfg.n_head", cfg.n_head)
        print("n_layer: cfg.n_layer", cfg.n_layer)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(0.1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.n_embd,
            nhead=cfg.n_head,
            dim_feedforward=4 * cfg.n_embd,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layer)
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.block_size = cfg.block_size
        self.cfg = cfg

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.block_size, "Sequence too long"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        # build causal mask: (T, T) with True at positions that should be masked
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=idx.device), 1)
        x = self.transformer(x, mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        for _ in range(max_new_tokens):
            if idx.size(1) > self.block_size:
                idx = idx[:, -self.block_size :]
            logits = self(idx)
            next_token = torch.distributions.Categorical(logits[:, -1, :].softmax(-1)).sample()
            idx = torch.cat([idx, next_token.unsqueeze(-1)], dim=1)
        return idx

def train(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)

    tok = CharTokenizer()
    ds = PokerPHHDataset(args.data_dir, tok, block_size=args.block_size)

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
    )
    model = GPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    for step, (x, y) in enumerate(dl): 
        print("x: ", x)
        print("y: ", y)

        # test tokenizer 
        print("x: ", tok.decode(x[0].tolist()))
        print("y: ", tok.decode(y[0].tolist()))
        break

    # test tokenizer 

    model.train()
    for epoch in range(1, args.epochs + 10):
        total_loss = 0.0
        for step, (x, y) in enumerate(dl):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (step + 1) % 100 == 0:
                avg = total_loss / 100
                print(f"Epoch {epoch} | Step {step+1}/{len(dl)} | loss {avg:.3f}")
                total_loss = 0.0
        # save checkpoint each epoch
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "vocab": tok.itos}, Path(args.save_dir)/f"ckpt_{epoch}.pt")

    print("Training complete.  Final model saved in", args.save_dir)



def get_args():
    p = argparse.ArgumentParser(description="Pre-train a GPT on PHH poker hands")
    p.add_argument("--data_dir", required=True, help="Directory with PHH txt files")
    p.add_argument("--save_dir", default="./ckpt", help="Where to save checkpoints")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_embd", type=int, default=512)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args)
