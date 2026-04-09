# evaluate_realworldqa.py
import re
from datasets import load_dataset
from model import load_model, run_inference
from methods.stratified_eviction import get_press

model, processor = load_model()
ds = load_dataset("xai-org/RealworldQA", split="test")
press = get_press(compression_ratio=0.5, vision_weight=0.2)

correct_count    = 0
total            = 0
total_prefill    = 0
total_decode     = 0
total_mem        = 0
total_throughput = 0
total_img_tokens = 0

def extract_letter(text):
    match = re.search(r'\b([A-E])\b', text.upper())
    return match.group(1) if match else text[0].upper() if text else ""

def is_correct(prediction, correct):
    # if correct answer is a single letter
    if len(correct) == 1:
        return extract_letter(prediction) == correct.upper()
    # if correct answer is a word (Yes/No, color, etc.)
    else:
        return correct.lower() in prediction.lower()

for sample in ds:
    if total >= 100:
        break
    image    = sample["image"]
    question = sample["question"]
    correct  = sample["answer"]

    prediction, metrics = run_inference(model, processor, image, question, max_new_tokens=16, press=press)
    predicted_letter = extract_letter(prediction)

    total_prefill    += metrics["prefill_ms"]
    total_decode     += metrics["decode_ms"]
    total_mem        += metrics["peak_mem_gb"]
    total_throughput += metrics["throughput_tokps"]
    total_img_tokens += metrics["num_image_tokens"]

    is_correct_flag = is_correct(prediction, correct)
    correct_count += is_correct_flag
    total += 1

    print(f"[{total}] Predicted: {predicted_letter} | Correct: {correct} | {'✓' if is_correct_flag else '✗'}")

print(f"\nAccuracy          : {correct_count}/{total} = {correct_count/total*100:.1f}%")
print(f"Avg prefill       : {total_prefill/total:.1f} ms")
print(f"Avg decode        : {total_decode/total:.1f} ms")
print(f"Avg peak memory   : {total_mem/total:.2f} GB")
print(f"Avg throughput    : {total_throughput/total:.1f} tok/s")
print(f"Avg image tokens  : {total_img_tokens/total:.1f}")

with open("results_realworldqa_stratified_cr0.5_vw0.2.txt", "w") as f:
    f.write(f"Accuracy          : {correct_count}/{total} = {correct_count/total*100:.1f}%\n")
    f.write(f"Avg prefill       : {total_prefill/total:.1f} ms\n")
    f.write(f"Avg decode        : {total_decode/total:.1f} ms\n")
    f.write(f"Avg peak memory   : {total_mem/total:.2f} GB\n")
    f.write(f"Avg throughput    : {total_throughput/total:.1f} tok/s\n")
    f.write(f"Avg image tokens  : {total_img_tokens/total:.1f}\n")