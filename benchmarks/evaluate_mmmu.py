# evaluate_mmmu.py
import ast
import re
from datasets import load_dataset, concatenate_datasets
from model import load_model, run_inference

model, processor = load_model()

subjects = [
    'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory',
    'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science',
    'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics',
    'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 'Manage',
    'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy',
    'Physics', 'Psychology', 'Public_Health', 'Sociology'
]

all_splits = [load_dataset("MMMU/MMMU", subject, split="validation") for subject in subjects]
ds = concatenate_datasets(all_splits)
print(f"Total samples: {len(ds)}")

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

for sample in ds:
    image = sample["image_1"]
    if image is None:
        continue

    question = sample["question"].replace("<image 1>", "").strip()
    options  = ast.literal_eval(sample["options"])
    correct  = sample["answer"]

    lettered = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
    prompt   = f"{question}\n\n{lettered}\n\nAnswer with a single letter only."

    prediction, metrics = run_inference(model, processor, image, prompt, max_new_tokens=16)
    predicted_letter = extract_letter(prediction)

    total_prefill    += metrics["prefill_ms"]
    total_decode     += metrics["decode_ms"]
    total_mem        += metrics["peak_mem_gb"]
    total_throughput += metrics["throughput_tokps"]
    total_img_tokens += metrics["num_image_tokens"]

    is_correct = predicted_letter == correct.upper()
    correct_count += is_correct
    total += 1

    print(f"[{total}] Predicted: {predicted_letter} | Correct: {correct} | {'✓' if is_correct else '✗'}")

print(f"\nAccuracy          : {correct_count}/{total} = {correct_count/total*100:.1f}%")
print(f"Avg prefill       : {total_prefill/total:.1f} ms")
print(f"Avg decode        : {total_decode/total:.1f} ms")
print(f"Avg peak memory   : {total_mem/total:.2f} GB")
print(f"Avg throughput    : {total_throughput/total:.1f} tok/s")
print(f"Avg image tokens  : {total_img_tokens/total:.1f}")

with open("results_mmmu.txt", "w") as f:
    f.write(f"Accuracy          : {correct_count}/{total} = {correct_count/total*100:.1f}%\n")
    f.write(f"Avg prefill       : {total_prefill/total:.1f} ms\n")
    f.write(f"Avg decode        : {total_decode/total:.1f} ms\n")
    f.write(f"Avg peak memory   : {total_mem/total:.2f} GB\n")
    f.write(f"Avg throughput    : {total_throughput/total:.1f} tok/s\n")
    f.write(f"Avg image tokens  : {total_img_tokens/total:.1f}\n")