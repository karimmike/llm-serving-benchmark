"""Run a benchmark sweep and write one JSON per run to results/.

vLLM (server must already be running with the chosen model):
    python scripts/run_benchmark.py --engine vllm --model qwen-fp16
HF baseline (loads the model in-process; stop the vLLM server first):
    python scripts/run_benchmark.py --engine hf --model qwen-fp16
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.config import DEFAULT_CONFIG, load_config, server_url  # noqa: E402
from bench.metrics import summarize  # noqa: E402
from bench.workloads import load_workloads, make_prompts  # noqa: E402

WARMUP_SEED_OFFSET = 1000  # warm-up prompts never overlap measured prompts


def gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return "unknown"


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def vllm_server_version(cfg: dict) -> str | None:
    try:
        with urllib.request.urlopen(f"{server_url(cfg)}/version", timeout=5) as r:
            return json.load(r).get("version")
    except OSError:
        return None


def write_result(out_dir: Path, meta: dict, records) -> Path:
    payload = {
        "metadata": meta,
        "summary": summarize(records, meta["config"]["gpu_price_per_hour_usd"]),
        "requests": [r.to_dict() for r in records],
    }
    stamp = meta["timestamp"].replace(":", "").replace("-", "")
    name = f"{meta['engine']}_{meta['model_key']}_{meta['workload']}_c{meta['concurrency']}_{stamp}.json"
    path = out_dir / name
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    s = payload["summary"]
    ttft_p50 = s["ttft_s"]["p50"] or 0.0
    print(f"  -> {name}: {s['output_tokens_per_s']:.1f} tok/s, TTFT p50 {ttft_p50:.3f}s, "
          f"errors {s['num_errors']}/{s['num_requests']}")
    return path


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_vllm(args, cfg: dict, base_meta: dict, workloads: dict, out_dir: Path) -> None:
    from bench import loadgen

    sweep = cfg["sweep"]
    model_id = base_meta["model"]
    base_url = f"{server_url(cfg)}/v1"
    n = args.requests or sweep["requests_per_level"]
    levels = args.concurrency or sweep["concurrency"]
    base_meta["vllm_version"] = vllm_server_version(cfg) or package_version("vllm")

    for wl in workloads.values():
        prompts = make_prompts(wl, n, sweep["seed"])
        warmup = make_prompts(wl, sweep["warmup_requests"], sweep["seed"] + WARMUP_SEED_OFFSET)
        for c in levels:
            print(f"[vllm] {args.model} {wl.name} concurrency={c} requests={n}")
            loadgen.run(base_url, model_id, warmup, c, wl.max_tokens, sweep["ignore_eos"])
            records = loadgen.run(base_url, model_id, prompts, c, wl.max_tokens, sweep["ignore_eos"])
            meta = {**base_meta, "workload": wl.name, "concurrency": c,
                    "max_tokens": wl.max_tokens, "timestamp": now_utc()}
            write_result(out_dir, meta, records)


def run_hf(args, cfg: dict, base_meta: dict, workloads: dict, out_dir: Path) -> None:
    from bench import hf_baseline

    sweep, hf_cfg = cfg["sweep"], cfg["hf_baseline"]
    n = args.requests or hf_cfg["requests"]
    base_meta["vllm_version"] = None
    tokenizer, model = hf_baseline.load_model(base_meta["model"], base_meta["dtype"])

    for wl in workloads.values():
        prompts = make_prompts(wl, n, sweep["seed"])
        warmup = make_prompts(wl, hf_cfg["warmup_requests"], sweep["seed"] + WARMUP_SEED_OFFSET)
        print(f"[hf] {args.model} {wl.name} requests={n}")
        hf_baseline.run(tokenizer, model, warmup, wl.max_tokens, sweep["ignore_eos"])
        records = hf_baseline.run(tokenizer, model, prompts, wl.max_tokens, sweep["ignore_eos"])
        meta = {**base_meta, "workload": wl.name, "concurrency": 1,
                "max_tokens": wl.max_tokens, "timestamp": now_utc()}
        write_result(out_dir, meta, records)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["vllm", "hf"], required=True)
    ap.add_argument("--model", required=True, help="model key from the config, e.g. qwen-fp16")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--workloads", nargs="+", help="subset of workloads (default: all)")
    ap.add_argument("--concurrency", nargs="+", type=int, help="override concurrency levels (vllm only)")
    ap.add_argument("--requests", type=int, help="override number of measured requests per run")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = cfg["models"][args.model]
    workloads = load_workloads(cfg)
    if args.workloads:
        workloads = {k: workloads[k] for k in args.workloads}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_meta = {
        "engine": args.engine,
        "model_key": args.model,
        "model": model["hf_id"],
        "dtype": model["dtype"],
        "quantization": model.get("quantization"),
        "gpu": gpu_name(),
        "python": platform.python_version(),
        "packages": {p: package_version(p) for p in ("vllm", "torch", "transformers", "openai")},
        "config": cfg,
    }
    if args.engine == "vllm":
        run_vllm(args, cfg, base_meta, workloads, out_dir)
    else:
        run_hf(args, cfg, base_meta, workloads, out_dir)


if __name__ == "__main__":
    main()
