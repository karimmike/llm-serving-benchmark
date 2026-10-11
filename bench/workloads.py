"""Deterministic synthetic prompt sets.

Prompts are random English-like text drawn from a fixed vocabulary with a
seeded RNG, so every run (and every engine) sees exactly the same prompts.
Each prompt is different, so vLLM's prefix cache only helps with the short
shared instruction, not the body.

Lengths are approximate: TOKENS_PER_WORD was calibrated with the
Qwen2.5 tokenizer. The real prompt token count (including the chat template)
is recorded per request from the server's usage report.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

INSTRUCTION = "Read the following notes and write a detailed summary of them.\n\n"

TOKENS_PER_WORD = 1.09  # measured with the Qwen2.5 tokenizer on this vocabulary

VOCAB = (
    "time year people way day man thing woman life child world school state family "
    "student group country problem hand part place case week company system program "
    "question work government number night point home water room mother area money "
    "story fact month lot right study book eye job word business issue side kind head "
    "house service friend father power hour game line end member law car city community "
    "name president team minute idea kid body information back parent face others level "
    "office door health person art war history party result change morning reason "
    "research girl guy moment air teacher force education foot boy age policy process "
    "music market sense nation plan college interest death experience effect class "
    "control care field development role effort rate heart drug show leader light voice "
    "wife police mind price report decision son view relationship town road arm "
    "difference value building action model season society tax director position player "
    "record paper space ground form event official matter center couple site project "
    "activity star table need court oil situation cost industry figure street image "
    "phone data picture practice piece land product doctor wall patient worker news test "
    "movie north love support technology step baby computer type attention film tree "
    "source organization hair window evidence population truth song answer network "
    "makes takes finds gives builds keeps brings holds turns starts shows hears plays "
    "runs moves lives believes happens writes provides sits stands loses pays meets "
    "includes continues sets learns leads understands watches follows stops creates "
    "speaks reads allows adds spends grows opens walks wins offers remembers considers "
    "new good high old great big small large local social important national young "
    "different early public bad able late hard major better economic strong possible "
    "whole free military true federal international full special easy clear recent "
    "certain personal open red difficult available likely short single medical current "
    "wrong private past foreign fine common poor natural significant similar hot dead "
    "central happy serious ready simple left physical general environmental financial"
).split()


@dataclass(frozen=True)
class Workload:
    name: str
    prompt_tokens: int  # approximate target length of the prompt body
    max_tokens: int  # output length (fixed when ignore_eos is on)


def _sentence(rng: random.Random) -> str:
    words = rng.choices(VOCAB, k=rng.randint(8, 16))
    return " ".join(words).capitalize() + "."


def make_prompt(rng: random.Random, target_tokens: int) -> str:
    target_words = max(1, round(target_tokens / TOKENS_PER_WORD))
    sentences, n_words = [], 0
    while n_words < target_words:
        s = _sentence(rng)
        sentences.append(s)
        n_words += s.count(" ") + 1
    return INSTRUCTION + " ".join(sentences)


def make_prompts(workload: Workload, n: int, seed: int) -> list[str]:
    """n deterministic prompts for a workload. Same (workload, n, seed) -> same list."""
    rng = random.Random(f"{workload.name}:{seed}")
    return [make_prompt(rng, workload.prompt_tokens) for _ in range(n)]


def load_workloads(cfg: dict) -> dict[str, Workload]:
    return {
        name: Workload(name=name, prompt_tokens=w["prompt_tokens"], max_tokens=w["max_tokens"])
        for name, w in cfg["workloads"].items()
    }
