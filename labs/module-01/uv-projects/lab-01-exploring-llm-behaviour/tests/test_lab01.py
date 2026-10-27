"""Offline tests: the deterministic parts of Lab 01. No API key, model download or network needed."""

import pytest
import torch

from lab01 import part4_context
from lab01.part2_temperature import FALLBACK_LOGITS, FALLBACK_TOKENS, TEMPERATURES, entropy, next_token_table
from lab01.part3_hallucination import DOI_PATTERN, GroundedAnswer, load_probes, verify_grounding
from lab01.part4_context import NEEDLE, ContextBudgetError, build_ops_log, preflight


def test_lower_temperature_sharpens_the_distribution():
    logits = torch.tensor(FALLBACK_LOGITS)
    probs = [torch.softmax(logits / t, dim=-1) for t in TEMPERATURES]
    assert all(abs(float(p.sum()) - 1.0) < 1e-4 for p in probs), "probabilities must sum to 1"
    top1 = [float(p.max()) for p in probs]
    assert top1 == sorted(top1, reverse=True), "a lower temperature must give the top token more probability"
    entropies = [entropy(p) for p in probs]
    assert entropies == sorted(entropies), "a higher temperature must spread the distribution"


def test_next_token_table_has_one_column_per_temperature():
    table = next_token_table(torch.tensor(FALLBACK_LOGITS), TEMPERATURES, lambda i: FALLBACK_TOKENS[i], k=3)
    assert list(table.columns) == [f"T={t}" for t in TEMPERATURES]
    assert list(table.index) == [" reset", " the", " log"]


def test_probe_set_is_complete():
    probes = load_probes()
    assert len(probes) >= 5
    assert all({"id", "trigger", "prompt", "ground_truth"} <= set(p) for p in probes)


def test_doi_pattern_finds_identifiers():
    assert DOI_PATTERN.findall("see doi:10.1000/xyz123 and https://doi.org/10.1145/3292500") == [
        "10.1000/xyz123",
        "10.1145/3292500",
    ]


def test_grounding_verdicts():
    article = "Restart the VPN client.  Then sign in with your new password."
    abstained = GroundedAnswer(supported=False, answer=None, quote=None)
    quoted = GroundedAnswer(supported=True, answer="Restart it", quote="restart the VPN   client.")
    invented = GroundedAnswer(supported=True, answer="Call us", quote="Call 555-0100 out of hours.")
    assert verify_grounding(abstained, article) == "abstained"
    assert verify_grounding(quoted, article) == "grounded (quote found)"
    assert verify_grounding(invented, article) == "UNVERIFIED (quote not in article)"


def test_needle_position_and_ground_truth():
    log, counts = build_ops_log(100, NEEDLE, 0.5)
    lines = log.splitlines()
    assert lines[50] == NEEDLE, "the needle must sit at the requested position"
    assert counts["Network Ops"] == sum(" P1 [Network Ops] " in line for line in lines)


def test_preflight_refuses_an_oversized_request(monkeypatch):
    monkeypatch.setattr(part4_context, "count_tokens", lambda provider, text: len(text.split()))
    assert preflight("offline", "one two three", max_output_tokens=10, budget_tokens=20) == 13
    with pytest.raises(ContextBudgetError):
        preflight("offline", "one two three", max_output_tokens=10, budget_tokens=12)
