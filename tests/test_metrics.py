import math

import pytest

from bench.metrics import (
    RequestRecord,
    cost_per_million_tokens,
    latency_stats,
    percentile,
    run_duration,
    summarize,
)


def make_record(send, chunks, output_tokens=None, error=None):
    return RequestRecord(
        send_time=send,
        first_token_time=chunks[0] if chunks else None,
        chunk_times=list(chunks),
        end_time=chunks[-1] if chunks else send + 0.1,
        output_tokens=len(chunks) if output_tokens is None else output_tokens,
        error=error,
    )


def test_percentile_matches_linear_interpolation():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(xs, 0) == 1.0
    assert percentile(xs, 50) == 3.0
    assert percentile(xs, 100) == 5.0
    assert percentile(xs, 95) == pytest.approx(4.8)  # numpy.percentile gives 4.8
    assert percentile([4.0, 1.0, 3.0, 2.0], 50) == 2.5  # unsorted input, even length


def test_percentile_empty_and_single():
    assert percentile([], 50) is None
    assert percentile([7.0], 99) == 7.0


def test_latency_stats_keys_and_mean():
    stats = latency_stats([1.0, 2.0, 3.0])
    assert set(stats) == {"mean", "p50", "p95", "p99"}
    assert stats["mean"] == 2.0
    assert latency_stats([])["mean"] is None


def test_record_ttft_e2e_and_itl():
    r = make_record(send=1.0, chunks=[1.5, 1.6, 1.8, 2.1])
    assert r.ok
    assert r.ttft == pytest.approx(0.5)
    assert r.e2e == pytest.approx(1.1)
    assert r.inter_token_latencies == pytest.approx([0.1, 0.2, 0.3])


def test_failed_record_is_not_ok():
    assert not make_record(0.0, [], error="ConnectionError: boom").ok
    assert not make_record(0.0, [0.2, 0.3], error="timeout").ok


def test_run_duration_spans_first_send_to_last_end():
    records = [make_record(0.0, [0.5, 1.0]), make_record(0.2, [0.9, 2.0])]
    assert run_duration(records) == pytest.approx(2.0)
    assert run_duration([]) == 0.0


def test_cost_per_million_tokens():
    # 1000 tok/s = 3.6M tok/hour; $0.36/hour -> $0.10 per 1M tokens
    assert cost_per_million_tokens(0.36, 1000.0) == pytest.approx(0.10)
    assert cost_per_million_tokens(0.36, 0.0) is None


def test_summarize_aggregates_successes_and_errors():
    records = [
        make_record(0.0, [0.1, 0.2, 0.3, 0.4]),  # ttft 0.1, e2e 0.4, 4 tokens
        make_record(0.0, [0.3, 0.5, 0.7, 0.9, 1.0], output_tokens=5),  # ttft 0.3, e2e 1.0
        make_record(0.5, [], error="APIError: overloaded"),
    ]
    s = summarize(records, gpu_price_per_hour=0.36)

    assert s["num_requests"] == 3
    assert s["num_errors"] == 1
    assert s["error_rate"] == pytest.approx(1 / 3)
    assert s["duration_s"] == pytest.approx(1.0)
    assert s["output_tokens_total"] == 9  # errors don't count
    assert s["output_tokens_per_s"] == pytest.approx(9.0)
    assert s["requests_per_s"] == pytest.approx(2.0)
    assert s["ttft_s"]["p50"] == pytest.approx(0.2)
    assert s["e2e_s"]["p50"] == pytest.approx(0.7)
    # ITL is pooled over all gaps: 0.1 x3, 0.2 x3, 0.1 x1
    assert s["itl_s"]["mean"] == pytest.approx((0.1 * 4 + 0.2 * 3) / 7)
    assert s["cost_per_1m_output_tokens_usd"] == pytest.approx(0.36 / (9 * 3600) * 1e6)


def test_summarize_all_failed_has_no_latency_and_no_cost():
    s = summarize([make_record(0.0, [], error="x"), make_record(0.0, [], error="y")], 0.35)
    assert s["error_rate"] == 1.0
    assert s["output_tokens_per_s"] == 0.0
    assert s["ttft_s"]["p50"] is None
    assert s["cost_per_1m_output_tokens_usd"] is None


def test_summarize_is_json_safe():
    s = summarize([make_record(0.0, [0.1, 0.2])], 0.35)
    flat = [v for v in s.values() if isinstance(v, float)]
    flat += [v for d in s.values() if isinstance(d, dict) for v in d.values() if v is not None]
    assert all(math.isfinite(v) for v in flat)
