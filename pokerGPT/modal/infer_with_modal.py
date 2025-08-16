#!/usr/bin/env python
import os
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
import modal

# Modal App
app = modal.App("poker-gpt-inference")

# Volume and image setup
vol = modal.Volume.from_name("pokerGPTTSMA")
# image = (
#     modal.Image.from_registry("pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel")
#     .pip_install_from_requirements("requirements.txt")
# )

image = modal.Image.from_registry("pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel").pip_install(
    "transformers>=4.41.0",
    "datasets>=2.19.0",
    "tokenizers>=0.15.2",
    "accelerate>=0.31.0",
    "torch>=2.2.0",
    "numpy>=1.24.0",
    "pyyaml>=6.0.1",
    "tqdm>=4.66.0",
    "matplotlib==3.10.3",
    "scikit-learn==1.7.0",
    "seaborn==0.13.2",
    "modal==1.1.2"
)



# Modal function
@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60 * 6,
    cpu=4,
    memory=24 * 1024,
    volumes = {"/data": vol}
)
def infer(ckpt: str, tokenizer_dir: str, context: str, max_new_tokens: int = 64,
          temp: float = 0.8, top_p: float = 0.95):

    # Load tokenizer
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(tokenizer_dir, "tokenizer.json"))
    tok.pad_token = "<PAD>"
    tok.eos_token = "<EOS>"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(ckpt)
    model.resize_token_embeddings(len(tok))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Build infilling prompt
    prompt = context.strip() + " <ANS>"
    ids = tok(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **ids,
            do_sample=True,
            temperature=temp,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.convert_tokens_to_ids("<EOS>")
        )

    gen = tok.decode(out[0], skip_special_tokens=False)
    # Extract only the part after <ANS> up to <EOS>
    ans = gen.split("<ANS>")[-1].split("<EOS>")[0].strip()
    print(ans)
    return ans

# Local entrypoint for testing
@app.local_entrypoint()
def main():
    # Example usage
    infer.remote(
        ckpt="/data/pokerGPT/artifacts/checkpoints/run1/best",
        tokenizer_dir="/data/pokerGPT/artifacts/tokenizer/",
        context='d dh p1 2hAc | d dh p2 8hQs | d dh p3 3cKc | d dh p4 8c7s | d dh p5 7dAs | d dh p6 Th8s | p3 f | p4 f | p5 cc | p6 cc | p1 cc | p2 cc | d db 6dTs7h | p1 cc | p2 cc | p5 cc | d db 9h | p1 cc | p2 cc | p5 cc | d db 9d | p1 cc | p2 cbr 640 | p5 cc | p6 cbr 2795 | p1 f | p2 cbr 9800 | p5 f | p6 cc | p2 sm 8hQs | p6 sm Th8s' ,
        max_new_tokens=40,
        temp=0.8,
        top_p=0.95
    )

if __name__ == "__main__":
    main()
