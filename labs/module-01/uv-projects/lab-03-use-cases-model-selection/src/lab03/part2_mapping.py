"""Part 2 — map each scenario to an agent type and model family, check it, and ask both providers to review it."""

from __future__ import annotations

import json
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from .common import ANTHROPIC_MODEL, OPENAI_MODEL, active_providers, claude_client, openai_client, show, step
from .part1_scenarios import CATALOGUE, SCENARIOS

AgentType = Literal[
    "single-call classifier", "extraction pipeline", "RAG assistant", "tool-using agent", "multi-agent orchestration"
]


class UseCaseMapping(BaseModel):
    scenario_id: str
    agent_type: AgentType
    model_tier: Literal["frontier", "balanced", "fast"]
    deployment: Literal["public API", "regional API", "self-hosted open weights"]
    candidates: list[str] = Field(min_length=2, description="model IDs from the catalogue, best first")
    rationale: str = Field(min_length=40)
    risks: list[str] = Field(min_length=1)


def team_mappings() -> list[UseCaseMapping]:
    """One worked mapping per scenario, UC-1 to UC-4 (the solution notebook's answers)."""
    return [
        UseCaseMapping(
            scenario_id="UC-1",
            agent_type="single-call classifier",
            model_tier="fast",
            deployment="regional API",
            candidates=["claude-haiku-5-5", "gpt-6-luna"],
            rationale="High volume, tiny outputs and a 2-second SLO: a fast tier with a structured-output schema "
            "is enough, and per-request cost dominates. No tools or loop needed.",
            risks=["misrouted P1s: escalate low-confidence labels to a human", "quality may differ by language"],
        ),
        UseCaseMapping(
            scenario_id="UC-2",
            agent_type="tool-using agent",
            model_tier="frontier",
            deployment="regional API",
            candidates=["claude-opus-5-5", "gpt-6-astra", "gpt-6.1-sol"],
            rationale="Multi-step reasoning over logs, CMDB and change records with tools; low volume, high stakes, "
            "so capability beats cost. Long inputs (up to 300K tokens) rule out the 131K open models.",
            risks=["wrong root cause stated confidently: require cited evidence", "tool misuse: human approves every action"],
        ),
        UseCaseMapping(
            scenario_id="UC-3",
            agent_type="extraction pipeline",
            model_tier="balanced",
            deployment="regional API",
            candidates=["claude-sonnet-5-5", "gpt-6.1-terra", "gpt-oss-120b"],
            rationale="A deterministic OCR → extraction → validation workflow run overnight through the Batch API. "
            "EU-only financial data: a regional endpoint, or self-hosted open weights behind our own OCR.",
            risks=["invented values for absent fields: nullable schema plus validation", "personal data in invoices"],
        ),
        UseCaseMapping(
            scenario_id="UC-4",
            agent_type="RAG assistant",
            model_tier="balanced",
            deployment="regional API",
            candidates=["claude-sonnet-5-5", "gpt-6.1-sol", "claude-haiku-5-5"],
            rationale="Answers must come from the KB with a citation, in six languages, within 6 seconds: retrieval "
            "plus a balanced model; multilingual embeddings for the retriever.",
            risks=["answers without a KB source: refuse or escalate", "weaker answers in Hindi and Japanese"],
        ),
    ]


def mapping_problems(mappings: list[UseCaseMapping]) -> list[str]:
    """Everything that makes the mappings inconsistent with the scenarios and the catalogue; empty means sound."""
    problems = []
    if sorted(m.scenario_id for m in mappings) != sorted(SCENARIOS):
        problems.append("map every scenario exactly once")
    for m in mappings:
        if m.scenario_id not in SCENARIOS:
            problems.append(f"{m.scenario_id}: not a scenario in data/use_case_scenarios.json")
            continue
        scenario = SCENARIOS[m.scenario_id]
        unknown = [c for c in m.candidates if c not in CATALOGUE]
        if unknown:
            problems.append(f"{m.scenario_id}: {unknown} not in data/model_catalogue.json")
            continue
        providers = {CATALOGUE[c]["provider"] for c in m.candidates if not CATALOGUE[c]["open_weight"]}
        if not {"openai", "anthropic"} <= providers:
            problems.append(f"{m.scenario_id}: name an OpenAI AND an Anthropic API model")
        for c in m.candidates:
            window = CATALOGUE[c]["max_input_tokens"] or CATALOGUE[c]["context_window"]
            if window < scenario["max_input_tokens"]:
                problems.append(
                    f"{m.scenario_id}: {c} holds {window:,} tokens but the scenario needs {scenario['max_input_tokens']:,}"
                )
        if m.deployment == "self-hosted open weights" and not any(CATALOGUE[c]["open_weight"] for c in m.candidates):
            problems.append(f"{m.scenario_id}: self-hosted needs an open-weight candidate")
    return problems


class MappingReview(BaseModel):
    agrees: bool = Field(description="True if the mapping is sound for production")
    biggest_risk: str = Field(description="The single most important risk the team should address")
    overlooked: list[str] = Field(description="Constraints in the scenario the rationale ignores (may be empty)")
    suggested_agent_type: AgentType | None = Field(description="A better agent type, or null if the team's is right")


REVIEW_SYSTEM = (
    "You are a principal architect reviewing an AI use-case mapping before an architecture board. "
    "Judge it against the scenario's volume, latency, residency, sensitivity and accuracy bar. Be concise."
)


def review_prompt(mapping: UseCaseMapping) -> str:
    catalogue = {
        c: {k: CATALOGUE[c][k] for k in ("tier", "deployment", "input", "output", "context_window")}
        for c in mapping.candidates
    }
    return (
        f"<scenario>\n{json.dumps(SCENARIOS[mapping.scenario_id], indent=1)}\n</scenario>\n"
        f"<mapping>\n{mapping.model_dump_json(indent=1)}\n</mapping>\n"
        f"<candidate_models>\n{json.dumps(catalogue, indent=1)}\n</candidate_models>"
    )


def review_openai(mapping: UseCaseMapping) -> MappingReview:
    response = openai_client().responses.parse(
        model=OPENAI_MODEL,
        instructions=REVIEW_SYSTEM,
        input=review_prompt(mapping),
        text_format=MappingReview,
        reasoning={"effort": "low"},
        max_output_tokens=3_000,
    )
    return response.output_parsed


def review_claude(mapping: UseCaseMapping) -> MappingReview:
    message = claude_client().messages.parse(
        model=ANTHROPIC_MODEL,
        max_tokens=3_000,
        system=REVIEW_SYSTEM,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": review_prompt(mapping)}],
        output_format=MappingReview,
    )
    return message.parsed_output


REVIEW = {"openai": review_openai, "anthropic": review_claude}


def run() -> None:
    mappings = team_mappings()
    step("The team's mappings")
    show(
        pd.DataFrame(
            [
                {
                    "scenario": m.scenario_id,
                    "agent type": m.agent_type,
                    "tier": m.model_tier,
                    "deployment": m.deployment,
                    "candidates": ", ".join(m.candidates),
                }
                for m in mappings
            ]
        ),
        index=False,
    )

    step("Checked against the scenarios and the catalogue")
    problems = mapping_problems(mappings)
    print("✅ every mapping is consistent" if not problems else "\n".join(f"❌ {p}" for p in problems))

    providers = active_providers()
    if not providers:
        print("ℹ️  The second-opinion review needs a model: skipping it.")
        return
    step("A second opinion from both providers: same prompt, same MappingReview schema")
    reviews = []
    for mapping in mappings:
        for provider in providers:
            review = REVIEW[provider](mapping)
            reviews.append({"scenario": mapping.scenario_id, "reviewer": provider, **review.model_dump()})
    show(pd.DataFrame(reviews), index=False, max_width=70)
