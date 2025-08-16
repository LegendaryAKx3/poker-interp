#!/usr/bin/env python
import argparse, os, json, re
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# ---------------------------
# Special tokens
# ---------------------------
SPECIALS = [
    "<PAD>", "<UNK>", "<BOS>", "<EOS>", "<GAP>", "<ANS>",
    "<SEP>",
    # roles/streets
    "d", "dh", "db",
    "p1", "p2", "p3", "p4", "p5", "p6",
    # actions
    "cc", "f", "cbr", "sm",
]

# Cards
RANKS = list("23456789TJQKA")
SUITS = list("cdhs")

CARD_RE = re.compile(r'([2-9TJQKA][cdhs])')

# ---------------------------
# Training data iterator
# ---------------------------
def iter_training_tokens(data_dir):
    for name in os.listdir(data_dir):
        if not name.endswith(".ndjson"):
            continue
        with open(os.path.join(data_dir, name), "r") as f:
            for line in f:
                try:
                    arr = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for item in arr:
                    tokens = item.strip().split()
                    for tok in tokens:
                        yield tok

# ---------------------------
# Main function
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Folder with *.ndjson hands")
    ap.add_argument("--out", required=True, help="Output folder for tokenizer.json")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Initialize BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = Whitespace()

    # BPE Trainer without bet bins
    trainer = BpeTrainer(
        vocab_size=5000,  # adjust as needed
        min_frequency=1,
        special_tokens=SPECIALS +
                       [r+s for r in RANKS for s in SUITS] +
                       ["|"]
    )

    # Train
    tokenizer.train_from_iterator(iter_training_tokens(args.data), trainer=trainer)

    # Save tokenizer
    tokenizer.save(os.path.join(args.out, "tokenizer.json"))
    print("✅ Saved BPE tokenizer to", os.path.join(args.out, "tokenizer.json"))

if __name__ == "__main__":
    main()
