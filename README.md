# llm-serving-benchmark

Why do serving engines matter? This project measures it on a free Kaggle T4:
plain Hugging Face `generate()` (one request at a time) versus vLLM under
concurrent load, using Qwen2.5-1.5B-Instruct in float16 and its 4-bit AWQ version.

## Goal

Quantify what a serving engine buys you on a small GPU:

- **Latency**: time to first token (TTFT), inter-token latency (ITL), end-to-end latency (p50/p95/p99)
- **Throughput**: output tokens/s and requests/s as concurrency grows (1, 4, 16, 64)
- **Cost**: $ per 1M output tokens, derived from throughput and an assumed GPU price
- **Quantization**: does a 4-bit AWQ model change the picture on a T4?

| | |
|---|---|
| GPU | Kaggle T4 (16 GB, sm_75): float16, no bfloat16, no FlashAttention |
| Engine | vLLM 0.18.0, OpenAI-compatible API, streaming |
| Models | [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), [`Qwen/Qwen2.5-1.5B-Instruct-AWQ`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-AWQ) |
| Workloads | short (~100 token) and long (~2,000 token) prompts, 128 output tokens |

## Setup

Local (client, analysis, tests; no GPU needed):

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
```

Kaggle: create a notebook with **GPU T4** and **Internet on**, then upload
`notebooks/kaggle_runner.ipynb` and set `REPO_URL` in its first cell.
It installs `requirements-kaggle.txt` (pinned versions).

## How to run

The notebook does all of this. To do it by hand on a GPU machine:

```bash
# 1. Start vLLM for one model (command built from configs/experiments.yaml)
python -c "from bench.config import *; print(' '.join(vllm_serve_command(load_config(), 'qwen-fp16')))"
vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype float16 ... > vllm.log 2>&1 &

# 2. Sweep concurrency levels x workloads -> results/*.json
python scripts/run_benchmark.py --engine vllm --model qwen-fp16

# 3. Stop the server, then run the HF baseline (same prompts, same output format)
python scripts/run_benchmark.py --engine hf --model qwen-fp16

# 4. Charts -> results/plots/
python scripts/plot_results.py
```

Each run writes one JSON with `metadata` (GPU, vLLM version, model, dtype,
full config, timestamp), `summary` (all metrics) and `requests` (raw per-request timings).

Repo layout:

```
bench/          loadgen.py (async client), hf_baseline.py, metrics.py, workloads.py, config.py
configs/        experiments.yaml (models, concurrency, workloads, GPU price)
scripts/        run_benchmark.py, plot_results.py
notebooks/      kaggle_runner.ipynb
tests/          CPU-only unit tests
results/        one JSON per run
```

## Results

_Not yet run._

## Findings

_Not yet run._

## Limitations

- **One small GPU, one small model.** A T4 and a 1.5B model; the gaps will look different on an A100/H100 or with larger models.
- **No FlashAttention on T4.** vLLM falls back to its other attention backends; HF uses PyTorch SDPA.
- **Closed-loop load.** A fixed number of requests in flight at all times, not an open-loop arrival process (Poisson), so this does not show queueing under bursty traffic.
- **Synthetic prompts.** Seeded random text from a fixed vocabulary; lengths are calibrated to the Qwen2.5 tokenizer and the true prompt token count is recorded per request.
- **Fixed output length.** `ignore_eos` (vLLM) and `min_new_tokens` (HF) force 128 output tokens so engines are compared like for like; real traffic has variable output lengths.
- **The HF baseline is deliberately naive**: one request at a time with no batching. It shows the cost of not having a serving engine, not the best HF can do.
- **The HF baseline runs only the float16 model.** AWQ is only benchmarked on vLLM.
- **Cost is an assumption.** Kaggle is free; `gpu_price_per_hour_usd` in the config is an assumed on-demand T4 price used to turn throughput into $ per 1M output tokens.
- **The client runs on the same machine as the server**, so it shares CPU with vLLM.
