from bench.workloads import INSTRUCTION, Workload, make_prompts

SHORT = Workload("short", prompt_tokens=100, max_tokens=64)
LONG = Workload("long", prompt_tokens=2000, max_tokens=64)


def test_prompts_are_deterministic():
    assert make_prompts(SHORT, 5, seed=0) == make_prompts(SHORT, 5, seed=0)
    assert make_prompts(SHORT, 5, seed=0) != make_prompts(SHORT, 5, seed=1)


def test_prompts_are_distinct_and_prefixed():
    prompts = make_prompts(SHORT, 10, seed=0)
    assert len(set(prompts)) == 10
    assert all(p.startswith(INSTRUCTION) for p in prompts)


def test_long_prompts_are_much_longer():
    short_words = len(make_prompts(SHORT, 1, seed=0)[0].split())
    long_words = len(make_prompts(LONG, 1, seed=0)[0].split())
    assert 15 < long_words / short_words < 25  # ~2000 vs ~100 tokens
