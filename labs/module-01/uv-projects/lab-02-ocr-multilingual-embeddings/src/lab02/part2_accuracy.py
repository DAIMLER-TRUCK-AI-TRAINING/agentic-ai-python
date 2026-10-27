"""Part 2 — measure OCR accuracy (CER), fix the layout, clean the scan, and extract invoice fields."""

from __future__ import annotations

import functools
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd
import pytesseract
from PIL import Image, ImageFilter, ImageOps
from pydantic import BaseModel, Field
from rapidfuzz.distance import Levenshtein

from .common import show, step
from .part1_tesseract import CLEAN_INVOICE, SCANNED_INVOICE, ground_truth, open_sample, tesseract_ok


def normalise_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def cer(predicted: str, reference: str) -> float:
    """Character error rate after collapsing whitespace: 0.0 is a perfect read."""
    reference = normalise_ws(reference)
    return Levenshtein.distance(normalise_ws(predicted), reference) / max(len(reference), 1)


def ocr(image: Image.Image, psm: int | None = None) -> str:
    """Run Tesseract; psm is the page segmentation mode (None = Tesseract's default, 3)."""
    return pytesseract.image_to_string(image, config=f"--psm {psm}" if psm else "")


def estimate_skew(image: Image.Image, limit: float = 5.0, step: float = 0.25) -> float:
    """Angle (degrees) that makes text rows most horizontal, by maximising the row-profile variance."""
    small = ImageOps.invert(image.convert("L")).resize((image.width // 3, image.height // 3))
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-limit, limit + step, step):
        rows = np.asarray(small.rotate(float(angle), resample=Image.Resampling.BILINEAR), dtype=np.float32).sum(axis=1)
        if (score := float(np.var(rows))) > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def otsu_threshold(pixels: np.ndarray) -> int:
    """The grey level that best separates ink from paper (Otsu's method)."""
    hist = np.bincount(pixels.ravel(), minlength=256).astype(np.float64)
    levels = np.arange(256)
    weight_bg = np.cumsum(hist)
    weight_fg = pixels.size - weight_bg
    mean_bg = np.cumsum(hist * levels) / np.maximum(weight_bg, 1)
    mean_fg = (np.sum(hist * levels) - np.cumsum(hist * levels)) / np.maximum(weight_fg, 1)
    return int(np.argmax(weight_bg * weight_fg * (mean_bg - mean_fg) ** 2))


def preprocess(image: Image.Image) -> tuple[Image.Image, float]:
    """Return (clean black-and-white image, skew angle corrected): grey, stretch, despeckle, deskew, binarise."""
    grey = ImageOps.autocontrast(ImageOps.grayscale(image), cutoff=1).filter(ImageFilter.MedianFilter(3))
    angle = estimate_skew(grey)
    grey = grey.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=255)
    pixels = np.asarray(grey)
    return Image.fromarray(((pixels > otsu_threshold(pixels)) * 255).astype(np.uint8)), angle


@functools.cache
def cer_table() -> pd.DataFrame:
    """CER of every sample: Tesseract defaults, psm 4, and preprocessing + psm 4."""
    rows = []
    for name, truth in ground_truth().items():
        image = open_sample(name)
        cleaned, angle = preprocess(image)
        rows.append(
            {
                "document": name,
                "default (psm 3)": cer(ocr(image), truth["text"]),
                "psm 4": cer(ocr(image, 4), truth["text"]),
                "preprocessed + psm 4": cer(ocr(cleaned, 4), truth["text"]),
                "skew corrected (°)": angle,
            }
        )
    return pd.DataFrame(rows).set_index("document")


class InvoiceFields(BaseModel):
    """The target schema, shared by the regex extractor and the vision models (Part 3)."""

    invoice_id: str | None = Field(description="Invoice number exactly as printed, e.g. INV-12345")
    invoice_date: str | None = Field(description="Invoice date as ISO 8601 YYYY-MM-DD")
    po_number: str | None = Field(description="Purchase-order number exactly as printed, or null if the document shows none")
    vendor_name: str | None = Field(description="The supplier's name as printed")
    currency: str | None = Field(description="ISO 4217 currency code of the total, e.g. EUR")
    total: str | None = Field(description="Grand total as digits with a decimal point and no separators, e.g. 2685.60")


def field_matches(field: str, predicted: str | None, expected: str | None) -> bool:
    """Compare one field; a blank prediction counts as None, an invented value never does."""
    predicted = (predicted or "").strip() or None
    if expected is None or predicted is None:
        return predicted is expected
    if field == "total":
        try:
            return Decimal(predicted.replace(",", "")) == Decimal(expected)
        except InvalidOperation:
            return False
    return predicted.casefold() == expected.casefold()


def field_report(predicted: InvoiceFields, expected: dict) -> dict[str, bool]:
    return {name: field_matches(name, getattr(predicted, name), value) for name, value in expected.items()}


def extract_invoice_fields(text: str) -> InvoiceFields:
    """Pull the six invoice fields out of OCR text with regular expressions; None when absent."""

    def find(pattern: str) -> str | None:
        match = re.search(pattern, text, flags=re.MULTILINE)
        return match.group(1).strip() if match else None

    raw_date = find(r"Invoice date:\s*(\d{1,2} [A-Za-z]{3} \d{4})")
    try:
        invoice_date = datetime.strptime(raw_date, "%d %b %Y").date().isoformat() if raw_date else None
    except ValueError:
        invoice_date = None
    total = re.search(r"Total\s*([A-Z]{3})\s+([\d,]+\.\d{2})", text)  # OCR often prints "TotalEUR"
    return InvoiceFields(
        invoice_id=find(r"\b(INV-\d{5})\b"),
        invoice_date=invoice_date,
        po_number=find(r"PO number:[ \t]*(PO-\d+)"),  # [ \t] so a blank field cannot run onto the next line
        vendor_name=find(r"^(.+?\bLtd)\b"),
        currency=total.group(1) if total else None,
        total=total.group(2).replace(",", "") if total else None,
    )


@functools.cache
def tesseract_fields() -> dict[str, InvoiceFields]:
    """Fields extracted from the preprocessed psm-4 text of both invoices."""
    fields = {}
    for name in (CLEAN_INVOICE, SCANNED_INVOICE):
        cleaned, _ = preprocess(open_sample(name))
        fields[name] = extract_invoice_fields(ocr(cleaned, 4))
    return fields


def run() -> None:
    if not tesseract_ok():
        print("❌ Tesseract is not installed: skipping this part (see labs/setup/README.md).")
        return
    step("Character error rate (CER) three ways: 0.0 is a perfect read")
    show(cer_table().round(3))

    step(f"Why {CLEAN_INVOICE} scores badly by default: reading order under psm 3 and psm 4 (lines 11-17)")
    clean = open_sample(CLEAN_INVOICE)
    for psm in (3, 4):
        lines = [line for line in ocr(clean, psm).splitlines() if line.strip()]
        print(f"--- psm {psm}\n" + "\n".join(lines[10:17]))

    step("Invoice fields by regex from the preprocessed psm-4 text (True = correct)")
    truth = ground_truth()
    show(pd.DataFrame({name: field_report(f, truth[name]["fields"]) for name, f in tesseract_fields().items()}))
