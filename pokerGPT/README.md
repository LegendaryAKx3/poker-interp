# Poker Hand-History LLM (NDJSON) — Starter Kit

This starter kit helps you train a powerful **decoder-only infilling model** (GPT-style) on
PHH/NDJSON hand histories like your examples. It includes:

- **Custom domain tokenizer** for cards/actions/roles/bets.
- **Preprocessing** from NDJSON to training sequences.
- **Infilling format**: turn a contiguous span of actions into a `<GAP>` and move
  the answer span to the end after `<ANS>` (prefix-LM style), so a causal model can learn to fill blanks.
- **Trainer** using Hugging Face Transformers (GPT-2 style by default).
- **Inference** to fill a blank deterministically or with sampling.
- **Activation extraction**: get hidden states / MLP activations for interpretability.

> Works with your strings like:
> `["d dh p1 2c6h", "p1 cc", "p2 f", "d db 4h8sTh", ...]`

## Quickstart (Colab / local)

```bash
# 1) Python 3.10+ recommended
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Put your NDJSON file(s) under data/
#   Each line is a JSON array of strings (one hand per line).
#   See data/example.ndjson for format.

# 3) Tokenizer: train or extend (optional)
python poker_tokenizer.py --data data --out artifacts/tokenizer

# 4) Preprocess to LM sequences (with infilling samples created on the fly):
python preprocess.py   --data data   --tokenizer artifacts/tokenizer   --out artifacts/dataset   --max-seq 1024   --mask-prob 0.5

# 5) Train
python train.py \
    --dataset artifacts/dataset \
    --tokenizer artifacts/tokenizer \
    --model gpt2 \
    --out artifacts/checkpoints/run1 \
    --epochs 1 \
    --batch 64 \
    --lr 5e-5 \
    --warmup 100 \
    --save-every 500 \
    --grad-accum 2 \
    --fp16 \
    --val-split 0.1 \
    --early-stop-patience 3


# 6) Inference (fill-in-the-blank)
python infer.py   --ckpt artifacts/checkpoints/run1   --tokenizer artifacts/tokenizer --context  'd dh p1 2hAc | <GAP> | d dh p3 3cKc | d dh p4 8c7s | d dh p5 7dAs | d dh p6 Th8s | p3 f | p4 f | p5 cc | p6 cc | p1 cc | p2 cc | d db 6dTs7h | p1 cc | p2 cc | p5 cc | d db 9h | p1 cc | p2 cc | p5 cc | d db 9d | p1 cc | p2 cbr 640 | p5 cc | p6 cbr 2795 | p1 f | p2 crb 9800 | p5 f | p6 cc | p2 sm 8hQs | p6 sm Th8s'   --max-new-tokens 40

# 7) Activation extraction
python activation_probe.py   --ckpt artifacts/checkpoints/run1   --tokenizer artifacts/tokenizer   --sequence 'd dh p1 2c6h | p1 cc | p2 cc | d db 4h8sTh | <GAP> | d db 7h'

# 8) Linear Probe
python linearProbe.py --model checkpoints/best                                                                                         
  --tokenizer artifacts/tokenizer/tokenizer.json                                                                                                                     
  --data data/hand3.ndjson
```
### Why this format? (Infilling with a causal model)
We transform a sequence like:
```
CTX_L  <GAP>  CTX_R
```
into a training sample like:
```
CTX_L  <GAP>  CTX_R  <ANS>  MISSING_SPAN  <EOS>
```
and compute loss **only** on tokens after `<ANS>`. The model learns to **output the missing span**
given the context on both sides.

### Bet-size normalization
Bet amounts (e.g. `cbr 9170`) are mapped to **bins** that encode a rough pot-fraction scale:
`BET_0`, `BET_1`, ... This avoids open-ended numeric vocab explosion and improves generalization.

### Extending power
- Use a larger base (`gpt2-medium`, `gpt2-xl`) or Llama-family via `AutoModelForCausalLM` (with their licenses).
- Scale context length (`--max-seq`), batch size (gradient accumulation), and tokens seen.
- Add **position/street embeddings** (dealer/seat, preflop/flop/turn/river) as learned special tokens.

---

## Data expectations

- **Input**: one hand per line, as JSON array of strings (NDJSON).
- The provided `example.ndjson` contains your three sample hands.

---

## Files

- `poker_tokenizer.py`: Build/extend a tokenizer with poker-specific specials.
- `preprocess.py`: Convert NDJSON to tokenized `.pt` shards.
- `train.py`: Fine-tune a GPT-like model on the preprocessed dataset with infilling objective.
- `infer.py`: Fill `<GAP>` given a context string.
- `activation_probe.py`: Extract hidden states / MLP activations.

Happy training!