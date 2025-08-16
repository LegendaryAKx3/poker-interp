#!/usr/bin/env python
import argparse, os, torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--context", required=True, help="Sequence with a single <GAP> and optional pipes '|'")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    args = ap.parse_args()

    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(args.tokenizer, "tokenizer.json"))
    tok.pad_token = "<PAD>"; tok.eos_token = "<EOS>"

    model = AutoModelForCausalLM.from_pretrained(args.ckpt)
    model.resize_token_embeddings(len(tok))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Build infilling prompt
    prompt = args.context.strip() + " <ANS>"
    ids = tok(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **ids,
            do_sample=True,
            temperature=args.temp,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.convert_tokens_to_ids("<EOS>")
        )

    gen = tok.decode(out[0], skip_special_tokens=False)
    # Extract only the part after <ANS> up to <EOS>
    ans = gen.split("<ANS>")[-1]
    ans = ans.split("<EOS>")[0].strip()
    print(ans)

if __name__ == "__main__":
    main()
