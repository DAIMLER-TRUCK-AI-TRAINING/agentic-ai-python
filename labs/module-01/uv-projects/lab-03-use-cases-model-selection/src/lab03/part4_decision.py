"""Part 4 — a weighted decision matrix for one scenario, a sensitivity check and the filled comparison sheet."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import pandas as pd

from .common import LAB_DIR, OUTPUT_DIR, show, step
from .part1_scenarios import CATALOGUE, CATALOGUE_DOC, SCENARIOS
from .part2_mapping import team_mappings
from .part3_cost import cost_or_none

TEMPLATE = LAB_DIR / "templates" / "model-comparison-sheet.md"
WEIGHTS = {
    "capability": 0.25,
    "latency": 0.10,
    "cost": 0.25,
    "residency": 0.20,
    "multilingual": 0.10,
    "operability": 0.10,
}
# Left for the team to fill in the exported sheet: these are decisions, not calculations.
DECISION = {
    "tie_breaker": "TODO: the criterion you agreed BEFORE scoring, e.g. the existing contract or a second provider",
    "trade_off": "TODO: the one trade-off you accepted, in one sentence",
    "review_date": "TODO: when you will re-run this decision (model tables change within months)",
    "rai_bias": "TODO: how you will measure quality per language or region",
    "rai_transparency": "TODO: how users know the output is AI-generated and see its sources",
    "rai_governance": "TODO: who owns the model choice, prompts and evals; what is logged",
}


def starter_ratings(model: dict, scenario: dict) -> dict[str, float]:
    """Rule-of-thumb ratings (1-5) for every criterion except cost. Change the rules if you disagree."""
    tier = model["tier"]
    hard = scenario["needs_tools"] or scenario["avg_input_tokens"] > 20_000
    capability = {"frontier": 5, "balanced": 4, "fast": 3 if hard else 4}[tier]
    latency = {"fast": 5, "balanced": 4, "frontier": 3}[tier]
    if scenario["latency_slo_seconds"] >= 600:
        latency = 5  # overnight batch: latency barely matters
    residency = 5 if model["deployment"] == "self-hosted" else 3
    multilingual = 4 if len(scenario["languages"]) > 1 and not model["open_weight"] else 3
    operability = 5 if model["deployment"] == "api" else 2
    return {
        "capability": capability,
        "latency": latency,
        "residency": residency,
        "multilingual": multilingual,
        "operability": operability,
    }


def cost_ratings(costs: dict[str, float]) -> dict[str, float]:
    """Map monthly costs to ratings: cheapest = 5.0, most expensive = 1.0, linear on a log10 scale.

    All costs equal → every model gets 5.0. Costs must be positive.
    """
    logs = {model: math.log10(cost) for model, cost in costs.items()}
    low, high = min(logs.values()), max(logs.values())
    if math.isclose(low, high):
        return {model: 5.0 for model in costs}
    return {model: round(5.0 - 4.0 * (value - low) / (high - low), 2) for model, value in logs.items()}


def weighted_score(ratings: dict[str, float], weights: dict[str, float]) -> float:
    """Sum of rating × weight. Weights must cover exactly the rated criteria and sum to 1.0."""
    if set(ratings) != set(weights):
        raise ValueError(f"criteria mismatch: {set(ratings) ^ set(weights)}")
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6):
        raise ValueError(f"weights must sum to 1.0, not {sum(weights.values())}")
    return round(sum(ratings[c] * weights[c] for c in weights), 2)


def decision_matrix(candidates: list[str], scenario: dict, weights: dict[str, float]) -> pd.DataFrame:
    """Ratings, monthly cost and weighted score per candidate, best first."""
    costs = {c: cost_or_none(CATALOGUE[c], scenario) for c in candidates}
    priced = {c: v for c, v in costs.items() if v}
    cost_rating = cost_ratings(priced) if priced else {}
    rows = {}
    for c in candidates:
        ratings = {**starter_ratings(CATALOGUE[c], scenario), "cost": cost_rating.get(c, 3.0)}  # unpriced → neutral 3
        rows[c] = {**ratings, "monthly $": costs[c], "score": weighted_score(ratings, weights)}
    return pd.DataFrame(rows).T.sort_values("score", ascending=False)


def renormalise(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def leaders(frame: pd.DataFrame) -> list[str]:
    """Every candidate that shares the top score: more than one means a tie."""
    return frame.index[frame.score == frame.score.max()].tolist()


def sensitivity(matrix: pd.DataFrame, candidates: list[str], scenario: dict) -> list[str]:
    """Does the winner survive a doubled cost weight, or a "self-hosted only" residency rule?"""
    notes = []
    top = leaders(matrix)
    if len(top) > 1:
        notes.append(f"Tie at {matrix.score.max()} between {', '.join(top)}: apply the tie-breaker you agreed before scoring.")
    alt = leaders(decision_matrix(candidates, scenario, renormalise({**WEIGHTS, "cost": WEIGHTS["cost"] * 2})))
    notes.append(f"Doubling the cost weight: leader(s) {'unchanged' if alt == top else 'change to ' + ', '.join(alt)} ({', '.join(alt)}).")
    self_hosted = [c for c in candidates if CATALOGUE[c]["deployment"] == "self-hosted"]
    notes.append(f"A self-hosted-only residency rule leaves: {self_hosted or 'no candidate: add an open-weight model or a regional API'}.")
    return notes


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(c) for c in frame.columns]
    lines = [
        "| " + " | ".join([frame.index.name or ""] + columns) + " |",
        "| " + " | ".join(["---"] * (len(columns) + 1)) + " |",
    ]
    for index, row in frame.iterrows():
        cells = ["—" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v) for v in row.to_list()]
        lines.append("| " + " | ".join([str(index)] + cells) + " |")
    return "\n".join(lines)


def export_sheet(
    scenario_id: str, matrix: pd.DataFrame, notes: list[str], *, team: str, output_dir: Path = OUTPUT_DIR
) -> Path:
    """Fill templates/model-comparison-sheet.md for one scenario and write it to `output_dir`."""
    s = SCENARIOS[scenario_id]
    mapping = next(m for m in team_mappings() if m.scenario_id == scenario_id)
    top = leaders(matrix)
    profile = pd.DataFrame(
        {
            "value": {
                k: s[k]
                for k in ("team", "requests_per_day", "avg_input_tokens", "avg_output_tokens",
                          "latency_slo_seconds", "residency", "data_sensitivity", "accuracy_bar")
            }
        }
    )
    profile.index.name = "field"
    candidates = pd.DataFrame(
        {
            c: {
                "provider": CATALOGUE[c]["provider"],
                "tier": CATALOGUE[c]["tier"],
                "deployment": CATALOGUE[c]["deployment"],
                "context window": f"{CATALOGUE[c]['context_window']:,}",
                "monthly $": cost_or_none(CATALOGUE[c], s),
            }
            for c in mapping.candidates
        }
    ).T
    candidates.index.name = "model"
    ranked = matrix.copy()
    ranked.index.name = "model"
    fields = {
        "scenario_id": scenario_id,
        "scenario_title": s["title"],
        "team": team,
        "date": date.today().isoformat(),
        "prices_as_of": CATALOGUE_DOC["prices_as_of"],
        "scenario_profile": markdown_table(profile),
        "agent_type": mapping.agent_type,
        "mapping_rationale": mapping.rationale,
        "model_tier": mapping.model_tier,
        "deployment": mapping.deployment,
        "candidates_table": markdown_table(candidates),
        "matrix_table": markdown_table(ranked),
        "sensitivity": "\n".join(f"- {n}" for n in notes),
        "decision": " / ".join(top) + (" (tie: see the tie-breaker)" if len(top) > 1 else ""),
        "fallback": next((m for m in ranked.index if m not in top), "—"),
        "risks": "\n".join(f"- {r}" for r in mapping.risks),
        **DECISION,
    }
    sheet = TEMPLATE.read_text(encoding="utf-8")
    for key, value in fields.items():
        sheet = sheet.replace("{{" + key + "}}", str(value))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"model-comparison-{scenario_id}.md"
    path.write_text(sheet, encoding="utf-8")
    return path


def run(scenario_id: str = "UC-1", team: str = "Team name") -> None:
    scenario = SCENARIOS[scenario_id]
    candidates = next(m.candidates for m in team_mappings() if m.scenario_id == scenario_id)

    step(f"{scenario_id} {scenario['title']}: starter ratings (1-5) and weights {WEIGHTS}")
    show(pd.DataFrame({c: starter_ratings(CATALOGUE[c], scenario) for c in candidates}).T)

    step("The weighted decision matrix, best first")
    matrix = decision_matrix(candidates, scenario, WEIGHTS)
    show(matrix)

    step("Sensitivity")
    notes = sensitivity(matrix, candidates, scenario)
    print("\n".join(notes))

    path = export_sheet(scenario_id, matrix, notes, team=team)
    print(f"\nWrote {path.relative_to(LAB_DIR)}: open it, then edit the TODO lines with your team.")
