"""Local code-LLM wrapper (MLX) used by the prompting baselines, SFT evaluation and GRPO."""

from __future__ import annotations

from dataclasses import dataclass

from spare.common.sandbox import strip_markdown

BASE_1_5B = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
BASE_7B = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"

SYSTEM = ("You are an expert mechanical designer who writes CadQuery (Python) programs for 3D-printable "
          "replacement parts. Reply with a single Python code block only.")

API_NOTES = """CadQuery reminders:
- Start with `import cadquery as cq`; only `cadquery` and `math` may be imported.
- Put each dimension in a named top-level variable (mm). Assign the final single solid to `result`.
- cq.Workplane("XY").box(x, y, z, centered=False) / .circle(r).extrude(h) / .rect(x, y).extrude(h)
- .faces(">Z").workplane().hole(d) ; .pushPoints([(x, y), ...]).circle(r).extrude(h) ; a.union(b) ; a.cut(b)
- .edges("|Z").fillet(r) ; .translate((x, y, z)) ; .rotate((0,0,0), (1,0,0), deg)
- cq.Solid.makeCylinder(r, h, cq.Vector(x, y, z), cq.Vector(dx, dy, dz)) wrapped by cq.Workplane("XY").add(...)
- Use the frame given: Z up is the print direction and the part sits on z >= 0."""


def build_messages(repr_text: str, few_shot: list[tuple[str, str]] | None = None, api_notes: bool = True) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM + ("\n\n" + API_NOTES if api_notes else "")}]
    for u, a in few_shot or []:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": f"```python\n{a.strip()}\n```"})
    msgs.append({"role": "user", "content": repr_text})
    return msgs


@dataclass
class GenConfig:
    temp: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1200
    batch_size: int = 16


class LLM:
    def __init__(self, model_id: str = BASE_1_5B, adapter_path: str | None = None):
        from mlx_lm import load

        self.model_id = model_id
        self.adapter_path = adapter_path
        self.model, self.tokenizer = load(model_id, adapter_path=adapter_path)

    def encode_chat(self, messages: list[dict], add_generation_prompt: bool = True) -> list[int]:
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt,
                                                  tokenize=True)

    def generate(self, prompts: list[list[int]], cfg: GenConfig, return_ids: bool = False):
        from mlx_lm import batch_generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=cfg.temp, top_p=cfg.top_p if cfg.temp > 0 else 0.0)
        texts, ids = [], []
        for i in range(0, len(prompts), cfg.batch_size):
            chunk = prompts[i:i + cfg.batch_size]
            r = batch_generate(self.model, self.tokenizer, chunk, max_tokens=cfg.max_tokens, sampler=sampler,
                               return_token_ids=return_ids, completion_batch_size=cfg.batch_size)
            texts += r.texts
            if return_ids:
                ids += r.token_ids
        return (texts, ids) if return_ids else texts


def extract_program(text: str) -> str:
    return strip_markdown(text).strip() + "\n"
