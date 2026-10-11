"""Async load generator for an OpenAI-compatible server (vLLM).

Closed loop: N requests, at most `concurrency` in flight at once (semaphore).
Each request streams, so we can timestamp the first token and every chunk.
"""

from __future__ import annotations

import asyncio
import time

from openai import AsyncOpenAI

from bench.metrics import RequestRecord


async def _send(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    model: str,
    prompt: str,
    max_tokens: int,
    ignore_eos: bool,
    t0: float,
) -> RequestRecord:
    async with sem:
        rec = RequestRecord(send_time=time.perf_counter() - t0)
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
                stream=True,
                stream_options={"include_usage": True},
                # vLLM extension: keep generating to max_tokens so every request
                # has the same output length (fair comparison with HF min_new_tokens).
                extra_body={"ignore_eos": ignore_eos},
            )
            async for chunk in stream:
                now = time.perf_counter() - t0
                if chunk.usage is not None:  # final chunk when include_usage=True
                    rec.prompt_tokens = chunk.usage.prompt_tokens
                    rec.output_tokens = chunk.usage.completion_tokens
                if chunk.choices and chunk.choices[0].delta.content:
                    if rec.first_token_time is None:
                        rec.first_token_time = now
                    rec.chunk_times.append(now)
            if rec.first_token_time is None:
                rec.error = "no tokens received"
            elif not rec.output_tokens:
                rec.output_tokens = len(rec.chunk_times)
        except Exception as e:  # record and keep going; error rate is a metric
            rec.error = f"{type(e).__name__}: {e}"
        finally:
            rec.end_time = time.perf_counter() - t0
        return rec


async def run_load(
    client: AsyncOpenAI,
    model: str,
    prompts: list[str],
    concurrency: int,
    max_tokens: int,
    ignore_eos: bool = True,
) -> list[RequestRecord]:
    sem = asyncio.Semaphore(concurrency)
    t0 = time.perf_counter()
    tasks = [_send(client, sem, model, p, max_tokens, ignore_eos, t0) for p in prompts]
    return await asyncio.gather(*tasks)


def run(
    base_url: str,
    model: str,
    prompts: list[str],
    concurrency: int,
    max_tokens: int,
    ignore_eos: bool = True,
    timeout_s: float = 600.0,
) -> list[RequestRecord]:
    """Synchronous entry point: one fresh client per run."""

    async def _main() -> list[RequestRecord]:
        async with AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout_s, max_retries=0) as client:
            return await run_load(client, model, prompts, concurrency, max_tokens, ignore_eos)

    return asyncio.run(_main())
