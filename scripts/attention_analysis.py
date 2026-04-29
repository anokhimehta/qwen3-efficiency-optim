"""
Attention Distribution Analysis
================================
Runs a sample of DocVQA inputs through Qwen3-VL and records the cumulative
attention scores per token, split by modality (visual vs text).

This visualizes WHY H2O is implicitly modality-aware: if text tokens
consistently receive higher cumulative attention than image tokens, H2O's
scoring naturally protects text tokens without an explicit modality constraint.

Usage (on HPC):
    python scripts/attention_analysis.py \
        --n_samples 50 \
        --output results/attention_distribution.json

Then plot locally:
    python scripts/plot_attention_distribution.py \
        --input results/attention_distribution.json \
        --output figures/fig5_attention_distribution.png
"""

import argparse
import json
import os
import torch
from datasets import load_dataset
from model import load_model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=50)
    p.add_argument("--output", default="results/attention_distribution.json")
    return p.parse_args()


def build_inputs(model, processor, image, prompt):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": prompt}
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    )
    inputs.pop("token_type_ids", None)
    return inputs.to(model.device)


def get_attention_scores(model, inputs):
    """
    Run one forward pass and collect per-token cumulative attention scores.
    Returns scores shape: (seq_len,) — sum of attention received across all
    heads and layers, which is exactly what H2O uses for eviction decisions.
    """
    seq_len = inputs["input_ids"].shape[1]
    # Accumulate attention scores across layers: shape (seq_len,)
    cumulative = torch.zeros(seq_len, device=model.device)

    hooks = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output is typically (hidden_states, attn_weights, ...)
            # attn_weights shape: (batch, num_heads, seq_len, seq_len)
            if isinstance(output, tuple) and len(output) > 1:
                attn = output[1]
                if attn is not None and attn.dim() == 4:
                    # Sum attention received by each token across heads
                    # attn[b, h, q, k] = attention from query q to key k
                    # We want: how much attention does each token k receive
                    received = attn[0].sum(dim=0).sum(dim=0)  # (seq_len,)
                    cumulative[:received.shape[0]] += received.detach().float()
        return hook

    # Register hooks on all attention layers
    for i, layer in enumerate(model.model.layers):
        h = layer.self_attn.register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        model(**inputs, output_attentions=True)

    for h in hooks:
        h.remove()

    return cumulative.cpu()


def main():
    args = parse_args()
    model, processor = load_model()

    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")

    results = []
    n = 0

    for sample in ds:
        if n >= args.n_samples:
            break

        image    = sample["image"].convert("RGB")
        question = sample["question"]
        prompt   = f"{question}\nAnswer concisely with just the value, no explanation."

        inputs = build_inputs(model, processor, image, prompt)
        seq_len = inputs["input_ids"].shape[1]

        # Find visual token positions
        num_image_tokens = (inputs["input_ids"] == model.config.image_token_id).sum().item()

        try:
            scores = get_attention_scores(model, inputs)
        except Exception as e:
            print(f"[{n+1}] Error: {e}, skipping")
            continue

        scores_list = scores.tolist()

        # Split by modality: visual tokens come first in Qwen3-VL
        visual_scores = scores_list[:num_image_tokens]
        text_scores   = scores_list[num_image_tokens:]

        results.append({
            "sample_id":        n,
            "seq_len":          seq_len,
            "num_image_tokens": num_image_tokens,
            "num_text_tokens":  seq_len - num_image_tokens,
            "visual_scores_mean": sum(visual_scores) / len(visual_scores) if visual_scores else 0,
            "text_scores_mean":   sum(text_scores)   / len(text_scores)   if text_scores   else 0,
            "visual_scores_max":  max(visual_scores) if visual_scores else 0,
            "text_scores_max":    max(text_scores)   if text_scores   else 0,
            # Store quantiles for distribution plotting
            "visual_scores": visual_scores[::max(1, len(visual_scores)//100)],  # subsample
            "text_scores":   text_scores[::max(1, len(text_scores)//20)],
        })

        n += 1
        print(f"[{n}/{args.n_samples}] seq_len={seq_len} "
              f"img_tokens={num_image_tokens} "
              f"visual_mean={results[-1]['visual_scores_mean']:.3f} "
              f"text_mean={results[-1]['text_scores_mean']:.3f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f)
    print(f"\nSaved {len(results)} samples to {args.output}")


if __name__ == "__main__":
    main()