"""Part 4 — a structured second opinion on each canvas from the small tier of each provider."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd
from llama_index.core import PromptTemplate
from pydantic import BaseModel, Field

from .common import needs_model, providers, rk, show, step
from .part3_canvases import CANVASES


class Critique(BaseModel):
    agrees: bool = Field(description="True if the decision is the best of the four options")
    strongest_counterargument: str = Field(description="One sentence")
    eval_to_run_first: str = Field(description="The single cheapest evaluation that would settle it")


CRITIC = PromptTemplate(
    "You review fine-tuning decisions. Options: prompt engineering, RAG, fine-tuning, hybrid. Note: the "
    "OpenAI fine-tuning API is being wound down and Claude cannot be fine-tuned, so fine-tuning means an "
    "open-weight model the company hosts.\n\nSCENARIO: {scenario}\nGAP: {gap}\nDECISION: {decision}\n"
    "WHY: {why}\nWOULD BE OVERTURNED BY: {overturn_eval}"
)


def run() -> None:
    if not needs_model("the critique"):
        return
    step(f"A structured critique per canvas: {', '.join(providers())} (small tier)")
    critiques = []
    for provider in providers():
        critic = rk.llama_llm(provider, small=True)
        for c in CANVASES:
            verdict = critic.structured_predict(Critique, CRITIC, **asdict(c))
            critiques.append({"critic": provider, "scenario": c.scenario[:30], "decision": c.decision, **verdict.model_dump()})
    critiques = pd.DataFrame(critiques)
    show(critiques[["critic", "scenario", "decision", "agrees", "strongest_counterargument"]], index=False, max_width=80)
    print(f"\n{'✅' if len(critiques) == len(CANVASES) * len(providers()) else '❌'} {len(critiques)} critiques returned a valid Critique")
    print("\n📝 Present one decision and the evaluation that would overturn it.")
