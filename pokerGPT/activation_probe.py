#!/usr/bin/env python
import argparse, os, torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--layer", type=int, default=-1, help="Which transformer layer to extract (-1 = last)")
    args = ap.parse_args()

    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(args.tokenizer, "tokenizer.json"))
    tok.pad_token = "<PAD>"; tok.eos_token = "<EOS>"
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, output_hidden_states=True)
    model.resize_token_embeddings(len(tok))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ids = tok(args.sequence, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model(**ids)
        hiddens = out.hidden_states  # tuple(layer+1)[batch, seq, hidden]
    layer_index = (len(hiddens)-1) if args.layer == -1 else args.layer
    acts = hiddens[layer_index][0].detach().cpu()  # [seq, hidden]
    # Save numpy for downstream analysis
    path = os.path.join(args.ckpt, "activations.pt")
    torch.save({"tokens": tok.convert_ids_to_tokens(ids["input_ids"][0]), "acts": acts}, path)
    print("Saved activations to", path, "shape", tuple(acts.shape))

if __name__ == "__main__":
    main()
