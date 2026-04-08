# evaluate_mathvista.py
import re
from datasets import load_dataset
from model import load_model, run_inference

model, processor = load_model()
ds = load_dataset("AI4Math/MathVista", split="testmini")

correct_count    = 0
total            = 0
total_prefill    = 0
total_decode     = 0
total_mem        = 0
total_throughput = 0
total_img_tokens = 0

def extract_number(text):
    matches = re.findall(r'-?\d+\.?\d*', text)
    return matches[-1] if matches else text.strip()

def is_correct(prediction, correct, question_type):
    if question_type == "multi_choice":
        return correct.lower() in prediction.lower()
    else:
        pred_num = extract_number(prediction)
        try:
            return abs(float(pred_num) - float(correct)) < 0.1
        except:
            return pred_num.strip() == correct.strip()

for sample in ds:
    image         = sample["decoded_image"]
    query         = sample["query"] + "\nGive only the final answer, nothing else."
    correct       = sample["answer"]
    question_type = sample["question_type"]

    prediction, metrics = run_inference(model, processor, image, query, max_new_tokens=64)

    total_prefill    += metrics["prefill_ms"]
    total_decode     += metrics["decode_ms"]
    total_mem        += metrics["peak_mem_gb"]
    total_throughput += metrics["throughput_tokps"]
    total_img_tokens += metrics["num_image_tokens"]

    correct_flag = is_correct(prediction, correct, question_type)
    correct_count += correct_flag
    total += 1

    print(f"[{total}] Type: {question_type}")
    print(f"  Predicted : {prediction}")
    print(f"  Correct   : {correct}")
    print(f"  {'✓' if correct_flag else '✗'}")

print(f"\nAccuracy          : {correct_count}/{total} = {correct_count/total*100:.1f}%")
print(f"Avg prefill       : {total_prefill/total:.1f} ms")
print(f"Avg decode        : {total_decode/total:.1f} ms")
print(f"Avg peak memory   : {total_mem/total:.2f} GB")
print(f"Avg throughput    : {total_throughput/total:.1f} tok/s")
print(f"Avg image tokens  : {total_img_tokens/total:.1f}")

with open("results_mathvista.txt", "w") as f:
    f.write(f"Accuracy          : {correct_count}/{total} = {correct_count/total*100:.1f}%\n")
    f.write(f"Avg prefill       : {total_prefill/total:.1f} ms\n")
    f.write(f"Avg decode        : {total_decode/total:.1f} ms\n")
    f.write(f"Avg peak memory   : {total_mem/total:.2f} GB\n")
    f.write(f"Avg throughput    : {total_throughput/total:.1f} tok/s\n")
    f.write(f"Avg image tokens  : {total_img_tokens/total:.1f}\n")