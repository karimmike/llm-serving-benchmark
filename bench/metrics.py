"""Per-request records and the metrics computed from them.

Pure Python (no numpy) so it can be unit-tested on any CPU.

Definitions (all times in seconds, relative to the start of a run):
- TTFT (time to first token): first_token_time - send_time
- ITL (inter-token latency): gaps between consecutive streamed chunks,
  pooled across all successful requests
- E2E latency: end_time - send_time
- Throughput: output tokens (or requests) / wall-clock duration of the run
- Cost per 1M output tokens: GPU $/hour divided by tokens produced per hour.
  The GPU price is an ASSUMPTION from the config (Kaggle itself is free); it
  turns throughput into a comparable cost figure, not a real bill.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field


@dataclass
class RequestRecord:
    send_time: float
    first_token_time: float | None = None
    chunk_times: list[float] = field(default_factory=list)
    end_time: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.first_token_time is not None and self.end_time is not None

    @property
    def ttft(self) -> float:
        return self.first_token_time - self.send_time

    @property
    def e2e(self) -> float:
        return self.end_time - self.send_time

    @property
    def inter_token_latencies(self) -> list[float]:
        t = self.chunk_times
        return [b - a for a, b in zip(t, t[1:])]

    def to_dict(self) -> dict:
        return asdict(self)


def percentile(values: list[float], q: float) -> float | None:
    """q-th percentile (0-100) with linear interpolation, same as numpy's default."""
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def latency_stats(values: list[float]) -> dict:
    return {
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def run_duration(records: list[RequestRecord]) -> float:
    """Wall-clock span from the first send to the last finished request."""
    if not records:
        return 0.0
    start = min(r.send_time for r in records)
    end = max(r.end_time if r.end_time is not None else r.send_time for r in records)
    return end - start


def cost_per_million_tokens(gpu_price_per_hour: float, tokens_per_second: float) -> float | None:
    if tokens_per_second <= 0:
        return None
    return gpu_price_per_hour / (tokens_per_second * 3600) * 1_000_000


def summarize(records: list[RequestRecord], gpu_price_per_hour: float) -> dict:
    ok = [r for r in records if r.ok]
    duration = run_duration(records)
    output_tokens = sum(r.output_tokens for r in ok)
    tokens_per_s = output_tokens / duration if duration > 0 else 0.0
    return {
        "num_requests": len(records),
        "num_errors": len(records) - len(ok),
        "error_rate": (len(records) - len(ok)) / len(records) if records else 0.0,
        "duration_s": duration,
        "requests_per_s": len(ok) / duration if duration > 0 else 0.0,
        "output_tokens_total": output_tokens,
        "output_tokens_per_s": tokens_per_s,
        "ttft_s": latency_stats([r.ttft for r in ok]),
        "itl_s": latency_stats([gap for r in ok for gap in r.inter_token_latencies]),
        "e2e_s": latency_stats([r.e2e for r in ok]),
        "gpu_price_per_hour_usd": gpu_price_per_hour,
        "cost_per_1m_output_tokens_usd": cost_per_million_tokens(gpu_price_per_hour, tokens_per_s),
    }
