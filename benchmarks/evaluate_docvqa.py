# evaluate_docvqa.py
import torch
from datasets import load_dataset
from model import load_model, run_inference

model, processor = load_model()
ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")

correct_count    = 0
total            = 0
total_prefill    = 0
total_decode     = 0
total_mem        = 0
total_throughput = 0
total_img_tokens = 0

for sample in ds:
    image    = sample["image"].convert("RGB")
    question = sample["question"]
    answers  = sample["answers"]

    prompt = f"{question}\nAnswer concisely with just the value, no explanation."

    prediction, metrics = run_inference(model, processor, image, prompt)

    total_prefill    += metrics["prefill_ms"]
    total_decode     += metrics["decode_ms"]
    total_mem        += metrics["peak_mem_gb"]
    total_throughput += metrics["throughput_tokps"]
    total_img_tokens += metrics["num_image_tokens"]

    is_correct = any(prediction.lower() == a.lower() for a in answers)
    correct_count += is_correct
    total += 1

    print(f"[{total}] Question : {question}")
    print(f"  Predicted     : {prediction}")
    print(f"  Accepted      : {answers}")
    print(f"  Prefill       : {metrics['prefill_ms']:.1f} ms")
    print(f"  Decode        : {metrics['decode_ms']:.1f} ms")
    print(f"  Peak memory   : {metrics['peak_mem_gb']:.2f} GB")
    print(f"  Throughput    : {metrics['throughput_tokps']:.1f} tok/s")
    print(f"  Image tokens  : {metrics['num_image_tokens']}")
    print(f"  {'✓' if is_correct else '✗'}")
    print()

print(f"Accuracy          : {correct_count}/{total} = {correct_count/total*100:.1f}%")
print(f"Avg prefill       : {total_prefill/total:.1f} ms")
print(f"Avg decode        : {total_decode/total:.1f} ms")
print(f"Avg peak memory   : {total_mem/total:.2f} GB")
print(f"Avg throughput    : {total_throughput/total:.1f} tok/s")
print(f"Avg image tokens  : {total_img_tokens/total:.1f}")

with open("results_docvqa.txt", "w") as f:
    f.write(f"Accuracy          : {correct_count}/{total} = {correct_count/total*100:.1f}%\n")
    f.write(f"Avg prefill       : {total_prefill/total:.1f} ms\n")
    f.write(f"Avg decode        : {total_decode/total:.1f} ms\n")
    f.write(f"Avg peak memory   : {total_mem/total:.2f} GB\n")
    f.write(f"Avg throughput    : {total_throughput/total:.1f} tok/s\n")
    f.write(f"Avg image tokens  : {total_img_tokens/total:.1f}\n")