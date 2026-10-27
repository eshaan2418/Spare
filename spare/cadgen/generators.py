"""LLM-backed program generators for the evaluation harness (MLX, local)."""

from __future__ import annotations

import gc

from spare.cadgen.evaluate import Generator, observe_all
from spare.cadgen.llm import LLM, GenConfig, build_messages, extract_program
from spare.common.provenance import read_jsonl
from spare.synth.registry import get as get_family

# Few-shot examples: fixed instances of TRAIN families (held-out instances), never test families.
FEW_SHOT_FAMILIES = ("battery_cover", "knob_dshaft")


def few_shot_examples(input_mode: str, seed: int = 0) -> list[tuple[str, str]]:
    recs = read_jsonl("data/benchmarks/test_id.jsonl")
    out = []
    for fam in FEW_SHOT_FAMILIES:
        r = next(x for x in recs if x["family"] == fam)
        assert get_family(fam).split == "train"
        x = observe_all([r], input_mode, seed)[0]
        out.append((x["repr"], get_family(fam).program(x["design_hat"], r["style"])))
    return out


_CACHE: dict[tuple[str, str | None], LLM] = {}
_SAMPLES: dict[tuple, list[str]] = {}


def get_llm(model_id: str, adapter: str | None = None) -> LLM:
    key = (model_id, adapter)
    if key not in _CACHE:
        for k in list(_CACHE):
            del _CACHE[k]
        gc.collect()
        _CACHE[key] = LLM(model_id, adapter)
    return _CACHE[key]


def llm_generator(model_id: str, adapter: str | None = None, few_shot: bool = False, api_notes: bool = True,
                  input_mode: str = "spare", temp: float = 0.7, max_tokens: int = 1024,
                  batch_size: int = 32) -> Generator:
    cfg_key = (model_id, adapter, few_shot, api_notes, input_mode, temp, max_tokens)

    def gen(recs, xs, n, seed):
        # samples are cached per (config, seed, uid) so first-sample and best-of-N systems share candidates
        todo = [i for i, r in enumerate(recs) if len(_SAMPLES.get((cfg_key, seed, r["uid"]), [])) < n]
        if todo:
            llm = get_llm(model_id, adapter)
            fs = few_shot_examples(input_mode, seed) if few_shot else None
            prompts = []
            for i in todo:
                p = llm.encode_chat(build_messages(xs[i]["repr"], fs, api_notes=api_notes))
                prompts += [p] * n
            import mlx.core as mx

            mx.random.seed(seed)
            texts = llm.generate(prompts, GenConfig(temp=temp, top_p=0.95, max_tokens=max_tokens,
                                                    batch_size=batch_size))
            progs = [extract_program(t) for t in texts]
            for j, i in enumerate(todo):
                _SAMPLES[(cfg_key, seed, recs[i]["uid"])] = progs[j * n:(j + 1) * n]
        return [_SAMPLES[(cfg_key, seed, r["uid"])][:n] for r in recs]
    return gen
