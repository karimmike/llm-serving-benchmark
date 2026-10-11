"""Config loading and the vLLM server command derived from it."""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "experiments.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def server_url(cfg: dict) -> str:
    s = cfg["server"]
    return f"http://{s['host']}:{s['port']}"


def vllm_serve_command(cfg: dict, model_key: str) -> list[str]:
    m, s = cfg["models"][model_key], cfg["server"]
    cmd = [
        "vllm", "serve", m["hf_id"],
        "--host", s["host"],
        "--port", str(s["port"]),
        "--dtype", m["dtype"],
        "--max-model-len", str(s["max_model_len"]),
        "--gpu-memory-utilization", str(s["gpu_memory_utilization"]),
        "--seed", str(cfg["sweep"]["seed"]),
    ]
    if m.get("quantization"):
        cmd += ["--quantization", m["quantization"]]
    return cmd
