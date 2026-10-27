"""Offline tests: the deterministic parts of Lab 02. No API key, model download or network needed."""

import numpy as np
import pandas as pd
import pytest

from lab02.part1_tesseract import CLEAN_INVOICE, SCANNED_INVOICE, ground_truth, tesseract_ok
from lab02.part2_accuracy import InvoiceFields, cer, extract_invoice_fields, field_matches, field_report, otsu_threshold
from lab02.part2_accuracy import tesseract_fields
from lab02.part5_similarity import word_overlap
from lab02.part6_matching import MATCH_COLUMNS, match_tickets


def test_cer_ignores_whitespace_and_counts_edits():
    assert cer("INV-88341  total\n2685.60", "INV-88341 total 2685.60") == 0.0
    assert cer("abcd", "abce") == pytest.approx(0.25)


def test_otsu_threshold_separates_ink_from_paper():
    pixels = np.array([20] * 500 + [230] * 500, dtype=np.uint8)
    assert 20 <= otsu_threshold(pixels) < 230


def test_extractor_meets_its_contract():
    sample = "Acme Parts Ltd\nInvoice number: INV-12345\nInvoice date: 02 Jan 2026\nPO number:\nTotalEUR 1,234.50"
    fields = extract_invoice_fields(sample)
    assert isinstance(fields, InvoiceFields)
    assert (fields.invoice_id, fields.invoice_date, fields.po_number) == ("INV-12345", "2026-01-02", None)
    assert (fields.vendor_name, fields.currency, fields.total) == ("Acme Parts Ltd", "EUR", "1234.50")


def test_field_matching_never_rewards_an_invented_value():
    assert field_matches("po_number", "  ", None), "a blank prediction counts as None"
    assert not field_matches("po_number", "PO-0000", None), "an invented PO number is wrong"
    assert field_matches("total", "2,685.60", "2685.60")
    assert not field_matches("total", "n/a", "2685.60")
    assert field_matches("vendor_name", "BLUEFIN NETWORK SUPPLIES LTD", "Bluefin Network Supplies Ltd")


def test_ground_truth_covers_the_four_samples():
    truth = ground_truth()  # generates data/samples/ on first use
    assert len(truth) == 4 and truth[SCANNED_INVOICE]["fields"]["po_number"] is None


@pytest.mark.skipif(not tesseract_ok(), reason="Tesseract is not installed")
def test_tesseract_pipeline_reads_both_invoices():
    truth = ground_truth()
    for name in (CLEAN_INVOICE, SCANNED_INVOICE):
        report = field_report(tesseract_fields()[name], truth[name]["fields"])
        assert all(report.values()), f"{name}: {report}"


def test_word_overlap_is_a_jaccard_score():
    assert word_overlap("the VPN is down", "the vpn is down") == 1.0
    assert word_overlap("VPN down", "printer jam") == 0.0


class KeywordEncoder:
    """A stand-in for SentenceTransformer: one dimension per keyword, normalised."""

    KEYWORDS = ("vpn", "printer")

    def encode(self, texts, normalize_embeddings=True):
        vectors = np.array([[float(k in t.lower()) for k in self.KEYWORDS] for t in texts]) + 1e-6
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_match_tickets_meets_its_contract():
    kb = pd.DataFrame(
        {"kb_id": ["KB-1", "KB-2"], "title": ["VPN access", "Printer queue"], "summary": ["Fix it.", "Clear it."]}
    )
    tickets = pd.DataFrame(
        {
            "ticket_id": ["INC-1", "INC-2", "INC-3"],
            "language": ["en", "de", "fr"],
            "gold_kb": ["KB-1", "KB-2", "KB-2"],
            "text": ["VPN rejects me", "Drucker: printer offline", "le VPN ne marche pas"],
        }
    )
    result = match_tickets(KeywordEncoder(), tickets, kb)
    assert list(result.columns) == MATCH_COLUMNS
    assert result.predicted_kb.tolist() == ["KB-1", "KB-2", "KB-1"]
    assert result.correct.tolist() == [True, True, False]
    assert result.score.between(-1, 1).all()
