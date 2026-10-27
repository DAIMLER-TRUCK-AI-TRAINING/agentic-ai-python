"""Offline tests: the deterministic parts of Lab 03. No API key or network needed."""

import math

import pytest
from pydantic import ValidationError

from lab03.part1_scenarios import CATALOGUE, SCENARIOS
from lab03.part2_mapping import UseCaseMapping, mapping_problems, team_mappings
from lab03.part3_cost import monthly_cost
from lab03.part4_decision import (
    WEIGHTS,
    cost_ratings,
    decision_matrix,
    export_sheet,
    leaders,
    sensitivity,
    weighted_score,
)

TEST_MODEL = {
    "id": "test",
    "input": 1.0,
    "cached_input": 0.1,
    "output": 5.0,
    "long_context": {"threshold_tokens": 1_500, "input": 2.0, "cached_input": 0.2, "output": 7.5},
}
TEST_SCENARIO = {"requests_per_day": 1_000, "avg_input_tokens": 1_000, "avg_output_tokens": 500}


def test_monthly_cost_against_hand_calculations():
    # 1,000 in × $1 + 500 out × $5 = $0.0035 per request × 1,000 × 30 = $105.00
    assert monthly_cost(TEST_MODEL, TEST_SCENARIO) == 105.00
    # half the input cached: 500 × $1 + 500 × $0.10 + 500 × $5 = $0.00305 → $91.50
    assert monthly_cost(TEST_MODEL, TEST_SCENARIO, cache_hit_rate=0.5) == 91.50
    assert monthly_cost(TEST_MODEL, TEST_SCENARIO, batch=True) == 52.50
    # 2,000 input tokens is over the 1,500 threshold: 2,000 × $2 + 500 × $7.5 = $0.00775 → $232.50
    assert monthly_cost(TEST_MODEL, {**TEST_SCENARIO, "avg_input_tokens": 2_000}) == 232.50


def test_open_weights_need_a_self_hosting_estimate():
    open_model = {"id": "open", "input": None}
    assert monthly_cost(open_model, TEST_SCENARIO, self_hosted_monthly_usd=4_000) == 4_000.00
    with pytest.raises(ValueError):
        monthly_cost(open_model, TEST_SCENARIO)


def test_cost_ratings_are_log_scaled():
    assert cost_ratings({"a": 10.0, "b": 100.0, "c": 1_000.0}) == {"a": 5.0, "b": 3.0, "c": 1.0}
    assert cost_ratings({"a": 42.0, "b": 42.0}) == {"a": 5.0, "b": 5.0}


def test_weighted_score_validates_its_weights():
    assert weighted_score({"x": 4, "y": 2}, {"x": 0.75, "y": 0.25}) == 3.5
    with pytest.raises(ValueError):
        weighted_score({"x": 4, "y": 2}, {"x": 0.5, "y": 0.4})
    with pytest.raises(ValueError):
        weighted_score({"x": 4}, {"x": 0.5, "y": 0.5})
    assert math.isclose(sum(WEIGHTS.values()), 1.0)


def test_team_mappings_are_consistent_with_the_catalogue():
    assert mapping_problems(team_mappings()) == []


def test_the_schema_and_the_checks_reject_a_weak_mapping():
    with pytest.raises(ValidationError):
        UseCaseMapping(
            scenario_id="UC-1", agent_type="single-call classifier", model_tier="fast",
            deployment="public API", candidates=["claude-haiku-5-5"], rationale="cheap", risks=[],
        )
    one_provider = UseCaseMapping(
        scenario_id="UC-1", agent_type="single-call classifier", model_tier="fast", deployment="public API",
        candidates=["claude-haiku-5-5", "claude-opus-5-5"],
        rationale="Two Claude models only, so the check must ask for an OpenAI one as well.",
        risks=["single vendor"],
    )
    problems = mapping_problems([one_provider])
    assert "map every scenario exactly once" in problems
    assert any("OpenAI AND an Anthropic" in p for p in problems)


def candidates_for(scenario_id):
    return next(m.candidates for m in team_mappings() if m.scenario_id == scenario_id)


def test_uc1_is_a_tie_as_the_guide_says():
    matrix = decision_matrix(candidates_for("UC-1"), SCENARIOS["UC-1"], WEIGHTS)
    assert sorted(leaders(matrix)) == ["claude-haiku-5-5", "gpt-6-luna"]
    assert matrix.score.max() == 4.25


def test_uc3_ranking():
    matrix = decision_matrix(candidates_for("UC-3"), SCENARIOS["UC-3"], WEIGHTS)
    assert list(matrix.index) == ["gpt-6.1-terra", "gpt-oss-120b", "claude-sonnet-5-5"]


def test_exported_sheet_fills_every_placeholder(tmp_path):
    matrix = decision_matrix(candidates_for("UC-1"), SCENARIOS["UC-1"], WEIGHTS)
    notes = sensitivity(matrix, candidates_for("UC-1"), SCENARIOS["UC-1"])
    path = export_sheet("UC-1", matrix, notes, team="Test team", output_dir=tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "{{" not in text and "Test team" in text
    assert all(c in CATALOGUE for c in candidates_for("UC-1"))
