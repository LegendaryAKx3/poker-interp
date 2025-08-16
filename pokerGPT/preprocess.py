#!/usr/bin/env python
import argparse, os, json, math, random, re, torch
from tokenizers import Tokenizer
from tqdm import tqdm

random.seed(1337)

BET_BINS = 64

def quantize_bet(value: int) -> str:
    v = max(1, int(value))
    x = math.log10(v) / 5.0
    b = max(0, min(BET_BINS-1, int(round(x * (BET_BINS-1)))))
    return f"BET_{b}"

def to_infilling_example(seq, mask_prob=0.5):
    if random.random() > mask_prob or len(seq) < 6:
        left, mid, right = seq, [], []
    else:
        k = random.randint(1, max(1, min(6, len(seq)//2)))
        i = random.randint(1, max(1, len(seq)-k-1))
        left = seq[:i]
        mid = seq[i:i+k]
        right = seq[i+k:]

    toks = left + ["<GAP>"] + right + ["<ANS>"] + (mid if mid else []) + ["<EOS>"]
    return toks, mid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--mask-prob", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok = Tokenizer.from_file(os.path.join(args.tokenizer, "tokenizer.json"))

    shard_inputs, shard_labels = [], []
    shard_idx = 0
    SHARD_SIZE = 2048

    for name in os.listdir(args.data):
        if not name.endswith(".ndjson"):
            continue
        with open(os.path.join(args.data, name)) as f:
            for line in tqdm(f, desc=f"Reading {name}"):
                try:
                    arr = json.loads(line)
                except:
                    continue

                seq = []
                for item in arr:
                    parts = item.strip().split()
                    # Normalize bets
                    if len(parts) == 3 and parts[1] == "cbr":
                        try:
                            parts[2] = quantize_bet(int(parts[2]))
                        except:
                            pass
                    seq += parts + ["|"]

                toks, missing_span = to_infilling_example(seq, mask_prob=args.mask_prob)

                # Tokenize whole sequence once
                ids = tok.encode(" ".join(toks)).ids

                # Build aligned labels
                labels_final = []
                after_ans = False
                for token in toks:
                    token_ids = tok.encode(token).ids
                    if token == "<ANS>":
                        after_ans = True
                        labels_final.extend([-100] * len(token_ids))
                    elif after_ans:
                        labels_final.extend(token_ids)
                    else:
                        labels_final.extend([-100] * len(token_ids))

                # Safety: match lengths
                if len(ids) != len(labels_final):
                    continue  # skip malformed sample

                # Truncate
                ids = ids[:args.max_seq]
                labels_final = labels_final[:args.max_seq]

                shard_inputs.append(torch.tensor(ids, dtype=torch.long))
                shard_labels.append(torch.tensor(labels_final, dtype=torch.long))

                if len(shard_inputs) >= SHARD_SIZE:
                    torch.save({"input_ids": shard_inputs, "labels": shard_labels},
                               os.path.join(args.out, f"shard_{shard_idx:05d}.pt"))
                    shard_idx += 1
                    shard_inputs, shard_labels = [], []

    if shard_inputs:
        torch.save({"input_ids": shard_inputs, "labels": shard_labels},
                   os.path.join(args.out, f"shard_{shard_idx:05d}.pt"))

    print("Wrote dataset shards to", args.out)

if __name__ == "__main__":
    main()
