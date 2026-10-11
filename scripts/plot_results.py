"""Plot latency and throughput vs concurrency from results/*.json.

    python scripts/plot_results.py [--results results] [--out results/plots]

One line per (engine, model). The HF baseline only has concurrency 1, so it
shows up as a single point. If a run was repeated, the newest file wins.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

LATENCY_METRICS = [("ttft_s", "TTFT"), ("itl_s", "Inter-token latency"), ("e2e_s", "End-to-end latency")]


def load_results(results_dir: Path) -> dict:
    """{workload: {series_label: {concurrency: summary}}}; newest run wins."""
    latest: dict = {}
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        m = data["metadata"]
        key = (m["workload"], f"{m['engine']} / {m['model_key']}", m["concurrency"])
        if key not in latest or m["timestamp"] >= latest[key][0]:
            latest[key] = (m["timestamp"], data["summary"])
    grouped: dict = defaultdict(lambda: defaultdict(dict))
    for (workload, label, c), (_, summary) in latest.items():
        grouped[workload][label][c] = summary
    return grouped


def _plot_lines(ax, series: dict, value, labeled: bool = True, **style) -> None:
    for label, by_c in sorted(series.items()):
        cs = sorted(by_c)
        ax.plot(cs, [value(by_c[c]) for c in cs], label=label if labeled else None, **style)


def _format(ax, title: str, ylabel: str) -> None:
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))  # 1, 4, 16, 64 instead of 2^k
    ax.set_xlabel("concurrency")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_workload(workload: str, series: dict, out_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (metric, title) in zip(axes, LATENCY_METRICS):
        _plot_lines(ax, series, lambda s, m=metric: s[m]["p50"], marker="o")
        ax.set_prop_cycle(None)  # same colors for the p95 lines
        _plot_lines(ax, series, lambda s, m=metric: s[m]["p95"], labeled=False, marker=".", linestyle="--")
        _format(ax, f"{title} (p50 solid, p95 dashed)", "seconds")
    axes[0].legend()
    fig.suptitle(f"Latency vs concurrency - {workload} prompts")
    fig.tight_layout()
    latency_path = out_dir / f"latency_{workload}.png"
    fig.savefig(latency_path, dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _plot_lines(axes[0], series, lambda s: s["output_tokens_per_s"], marker="o")
    _format(axes[0], "Output throughput", "tokens / s")
    _plot_lines(axes[1], series, lambda s: s["requests_per_s"], marker="o")
    _format(axes[1], "Request throughput", "requests / s")
    axes[0].legend()
    fig.suptitle(f"Throughput vs concurrency - {workload} prompts")
    fig.tight_layout()
    throughput_path = out_dir / f"throughput_{workload}.png"
    fig.savefig(throughput_path, dpi=120)
    plt.close(fig)
    return [latency_path, throughput_path]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "results" / "plots"))
    args = ap.parse_args()

    grouped = load_results(Path(args.results))
    if not grouped:
        raise SystemExit(f"no result files in {args.results}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for workload, series in sorted(grouped.items()):
        for path in plot_workload(workload, series, out_dir):
            print(path)


if __name__ == "__main__":
    main()
