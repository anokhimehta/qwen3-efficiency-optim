# model.py
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

def load_model():
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct",
        dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    model.eval()
    return model, processor

def run_inference(model, processor, image, prompt, max_new_tokens=64):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt}
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    )
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(model.device)

    # input sequence length
    input_seq_len = inputs["input_ids"].shape[1]

    # number of image tokens
    num_image_tokens = (inputs["input_ids"] == model.config.image_token_id).sum().item()

    # reset memory stats
    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    # prefill latency (max_new_tokens=1)
    start.record()
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    end.record()
    torch.cuda.synchronize()
    prefill_ms = start.elapsed_time(end)

    # full generation
    start.record()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    end.record()
    torch.cuda.synchronize()
    full_ms = start.elapsed_time(end)

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    decode_ms   = max(0.0, full_ms - prefill_ms)

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
    prediction  = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    num_output_tokens = len(trimmed[0])
    throughput = num_output_tokens / (full_ms / 1000)  # tok/s

    metrics = {
        "prefill_ms":        prefill_ms,
        "decode_ms":         decode_ms,
        "full_ms":           full_ms,
        "peak_mem_gb":       peak_mem_gb,
        "throughput_tokps":  throughput,
        "num_image_tokens":  num_image_tokens,
        "input_seq_len":     input_seq_len,
        "num_output_tokens": num_output_tokens,
    }

    return prediction, metrics