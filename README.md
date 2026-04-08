# qwen3-efficiency-optim

Benchmarking and KV cache optimization experiments on **Qwen3-VL-4B-Instruct**, a 4B-parameter vision-language model. The goal is to measure baseline inference efficiency across multimodal benchmarks and evaluate compression methods that reduce KV cache memory usage with minimal accuracy loss.

---

## Project Structure

```
.
├── model.py                        # Model loading and instrumented inference
├── methods/
│   ├── baseline.py                 # Minimal single-image inference example
│   ├── h2o.py                      # H2O KV cache eviction (via kvpress)
│   └── stratified-eviction.py      # Stratified eviction method (placeholder)
├── benchmarks/
│   ├── evaluate_docvqa.py          # DocVQA evaluation
│   ├── evaluate_mathvista.py       # MathVista evaluation
│   ├── evaluate_mmmu.py            # MMMU evaluation
│   └── evaluate_realworldqa.py     # RealWorldQA evaluation
├── scripts/
│   ├── run_docvqa.sh               # SLURM job script for DocVQA
│   ├── run_mathvista.sh            # SLURM job script for MathVista
│   ├── run_mmmu.sh                 # SLURM job script for MMMU
│   └── run_realworldqa.sh          # SLURM job script for RealWorldQA
└── baseline_results/
    ├── results_docvqa.txt
    ├── results_mathvista.txt
    ├── results_mmmu.txt
    └── results_realworldqa.txt
```

---

## How the Files Work Together

### `model.py` — Core inference engine

All benchmark scripts import from this file. It provides two functions:

- **`load_model()`** — loads `Qwen/Qwen3-VL-4B-Instruct` in `bfloat16` with `device_map="auto"`.
- **`run_inference(model, processor, image, prompt)`** — runs a two-pass inference to separately measure prefill and decode latency, and returns both the text prediction and a metrics dict with:
  - `prefill_ms`, `decode_ms`, `full_ms`
  - `peak_mem_gb` (GPU peak memory)
  - `throughput_tokps` (output tokens/second)
  - `num_image_tokens`, `input_seq_len`, `num_output_tokens`

### `methods/` — Compression strategies

Each file in `methods/` is a self-contained implementation of a KV cache compression technique that can be plugged into the inference pipeline.

| File | Method | Description |
|---|---|---|
| `baseline.py` | No compression | Minimal example of loading the model and running inference on a single image |
| `h2o.py` | H2O (Heavy Hitter Oracle) | Uses `kvpress.ExpectedAttentionPress` to evict low-attention KV cache entries at a configurable compression ratio |
| `stratified-eviction.py` | Stratified eviction | Placeholder for a custom eviction strategy |

### `benchmarks/` — Evaluation scripts

Each script loads the model via `model.py`, streams through a benchmark dataset, calls `run_inference` for each sample, accumulates accuracy and efficiency metrics, prints per-sample results, and writes a summary `results_<benchmark>.txt`.

| Script | Dataset | Task | Metric |
|---|---|---|---|
| `evaluate_docvqa.py` | `lmms-lab/DocVQA` (validation) | Document question answering | Exact match accuracy |
| `evaluate_mathvista.py` | `AI4Math/MathVista` (testmini) | Math reasoning from images | Multi-choice + numeric match |
| `evaluate_mmmu.py` | `MMMU/MMMU` (validation, 30 subjects) | Multi-discipline college-level QA | Letter-choice accuracy |
| `evaluate_realworldqa.py` | `xai-org/RealworldQA` (test) | Real-world visual reasoning | Letter/word match |

### `scripts/` — SLURM job launchers

These are batch submission scripts for NYU's HPC cluster (`greene`). Each script sets resource requirements (1 GPU, 40 GB RAM, up to 6 hours) and runs the corresponding benchmark inside a Singularity container with a cached HuggingFace model at `/scratch/am16455/hf_cache`.

To submit a job:
```bash
sbatch scripts/run_docvqa.sh
```

---

## Baseline Results

Measured on a single A100 GPU (g2-standard-12 partition) with no KV cache compression.

| Benchmark | Accuracy | Avg Prefill | Avg Decode | Peak Mem | Throughput | Avg Image Tokens |
|---|---|---|---|---|---|---|
| DocVQA | 87.8% (4699/5349) | 1649.3 ms | 279.3 ms | 9.94 GB | 3.8 tok/s | 3662.3 |
| MathVista | 35.9% (359/1000) | 222.6 ms | 296.6 ms | 9.03 GB | 12.8 tok/s | 429.7 |
| MMMU | 46.4% (418/900) | 216.4 ms | 174.4 ms | 9.05 GB | 11.4 tok/s | 464.2 |
| RealWorldQA | 71.6% (548/765) | 456.3 ms | 69.7 ms | 9.27 GB | 4.2 tok/s | 1296.0 |

DocVQA has significantly higher prefill latency and memory usage due to its much larger average image token count (~3662 vs ~430–1296 for others).

---

## Dependencies

- `torch`
- `transformers`
- `datasets`
- `Pillow`
- `kvpress` (for H2O method)
