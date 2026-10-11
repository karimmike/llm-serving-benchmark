"""Baseline: plain Hugging Face generate(), one request at a time.

TTFT is measured with a TextIteratorStreamer subclass that timestamps every
token step as generate() hands it over. Output records use the same
RequestRecord format as the vLLM load generator, so metrics are identical.

Needs torch + transformers + a GPU (Kaggle); not imported by the CPU tests.
"""

from __future__ import annotations

import time
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from bench.metrics import RequestRecord


class TimingStreamer(TextIteratorStreamer):
    """Records a timestamp each time generate() emits a new token."""

    def __init__(self, tokenizer, clock):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.clock = clock
        self.token_times: list[float] = []

    def put(self, value):
        if not self.next_tokens_are_prompt:  # first put() is the prompt itself
            self.token_times.append(self.clock())
        super().put(value)


def load_model(hf_id: str, dtype: str = "float16"):
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=getattr(torch, dtype), attn_implementation="sdpa"
    ).to("cuda")
    model.eval()
    return tokenizer, model


def generate_one(tokenizer, model, prompt: str, max_tokens: int, ignore_eos: bool, t0: float) -> RequestRecord:
    clock = lambda: time.perf_counter() - t0  # noqa: E731
    rec = RequestRecord(send_time=clock())
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        rec.prompt_tokens = inputs["input_ids"].shape[1]
        streamer = TimingStreamer(tokenizer, clock)
        result: dict = {}

        def _generate():
            try:
                with torch.inference_mode():
                    result["output"] = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        min_new_tokens=max_tokens if ignore_eos else 0,
                        do_sample=False,
                        temperature=None, top_p=None, top_k=None,  # silence sampling warnings
                        streamer=streamer,
                        pad_token_id=tokenizer.eos_token_id,
                    )
            except Exception as e:
                result["error"] = e
                streamer.end()  # unblock the consumer below

        thread = Thread(target=_generate)
        thread.start()
        for _ in streamer:  # drain decoded text; timing happens in put()
            pass
        thread.join()

        if "error" in result:
            raise result["error"]
        rec.chunk_times = streamer.token_times
        rec.first_token_time = rec.chunk_times[0] if rec.chunk_times else None
        rec.output_tokens = result["output"].shape[1] - rec.prompt_tokens
        if rec.first_token_time is None:
            rec.error = "no tokens received"
    except Exception as e:
        rec.error = f"{type(e).__name__}: {e}"
    finally:
        rec.end_time = clock()
    return rec


def run(tokenizer, model, prompts: list[str], max_tokens: int, ignore_eos: bool = True) -> list[RequestRecord]:
    t0 = time.perf_counter()
    return [generate_one(tokenizer, model, p, max_tokens, ignore_eos, t0) for p in prompts]
