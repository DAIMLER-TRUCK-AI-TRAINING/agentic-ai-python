#!/usr/bin/env python3
"""Generate the synthetic invoice and support-ticket images for Module 1 Lab 02.

    python labs/module-01/data/make_samples.py          # writes data/samples/
    python labs/module-01/data/make_samples.py --force  # regenerate

Every document is synthetic: the supplier, the reporters and every identifier
are invented for the course. The script also writes ``ground_truth.json`` with
the exact text and fields of each image, so OCR accuracy can be measured.
Output is deterministic: the same Pillow version produces the same pixels.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

OUT_DIR = Path(__file__).resolve().parent / "samples"
PAGE = (1240, 1754)  # A4 at 150 dpi
MARGIN = 90
FOOTER = "SYNTHETIC SAMPLE - TRAINING USE ONLY"


@dataclass
class Line:
    """One printed line: text, font size and optional right-aligned columns."""

    text: str
    size: int = 30
    columns: list[tuple[int, str]] = field(default_factory=list)  # (x position, text)
    gap: int = 14


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def render(lines: list[Line]) -> Image.Image:
    """Draw ``lines`` top to bottom on a white page."""
    page = Image.new("L", PAGE, color=255)
    draw = ImageDraw.Draw(page)
    y = MARGIN
    for line in lines:
        f = font(line.size)
        draw.text((MARGIN, y), line.text, fill=20, font=f)
        for x, text in line.columns:
            width = draw.textlength(text, font=f)
            draw.text((x - width, y), text, fill=20, font=f)  # right-align at x
        y += line.size + line.gap
    draw.text((MARGIN, PAGE[1] - MARGIN), FOOTER, fill=90, font=font(22))
    return page


def page_text(lines: list[Line]) -> str:
    """The ground-truth text of a page, one printed line per text line."""
    rows = []
    for line in lines:
        parts = [line.text] + [text for _, text in line.columns]
        row = "  ".join(p for p in parts if p)
        if row.strip():
            rows.append(row)
    rows.append(FOOTER)
    return "\n".join(rows)


def money(value: float) -> str:
    return f"{value:,.2f}"


def invoice_lines(
    number: str, date: str, due: str, po: str, items: list[tuple[str, int, float]]
) -> tuple[list[Line], str]:
    subtotal = round(sum(q * p for _, q, p in items), 2)
    vat = round(subtotal * 0.20, 2)
    total = round(subtotal + vat, 2)
    col_qty, col_unit, col_amount = 760, 960, 1150
    lines = [
        Line("INVOICE", size=56, gap=24),
        Line("Bluefin Network Supplies Ltd", size=34),
        Line("Unit 4, Harbour Trade Park, Example City EX1 2AB", size=24),
        Line("VAT reg. XX-0000000 (synthetic)", size=24, gap=36),
        Line(f"Invoice number: {number}"),
        Line(f"Invoice date: {date}"),
        Line(f"Due date: {due}"),
        Line(f"PO number: {po}".rstrip()),
        Line("Bill to: Enterprise IT Operations - Finance"),
        Line("Cost centre: CC-4410", gap=40),
        Line("Description", columns=[(col_qty, "Qty"), (col_unit, "Unit price"), (col_amount, "Amount")], gap=22),
    ]
    for description, qty, price in items:
        lines.append(
            Line(description, columns=[(col_qty, str(qty)), (col_unit, money(price)), (col_amount, money(qty * price))])
        )
    lines += [
        Line("", gap=10),
        Line("", columns=[(col_unit, "Subtotal"), (col_amount, money(subtotal))]),
        Line("", columns=[(col_unit, "VAT 20%"), (col_amount, money(vat))]),
        Line("", size=34, columns=[(col_unit, "Total EUR"), (col_amount, money(total))], gap=40),
        Line("Payment terms: 30 days net. Bank details on file.", size=26),
    ]
    return lines, f"{total:.2f}"


def ticket_lines(fields: dict[str, str], description: list[str]) -> list[Line]:
    lines = [Line("IT SERVICE DESK - INCIDENT RECORD", size=44, gap=30)]
    lines += [Line(f"{key}: {value}") for key, value in fields.items()]
    lines.append(Line("Description:", gap=10))
    lines += [Line(text, size=28, gap=8) for text in description]
    return lines


# --- degradations --------------------------------------------------------------


def degrade_scan(img: Image.Image, seed: int) -> Image.Image:
    """A tired office scanner: slight skew, blur, low contrast, speckle, JPEG artefacts."""
    rng = np.random.default_rng(seed)
    img = img.rotate(1.8, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=255)
    img = img.filter(ImageFilter.GaussianBlur(0.9))
    img = ImageEnhance.Contrast(img).enhance(0.55)
    pixels = np.asarray(img, dtype=np.int16)
    pixels = pixels + rng.normal(0, 14, pixels.shape).astype(np.int16)
    speckle = rng.random(pixels.shape) < 0.004
    pixels[speckle] = 40
    img = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    return jpeg_roundtrip(img, quality=35)


def degrade_photo(img: Image.Image, seed: int) -> Image.Image:
    """A phone photo of a printout: rotation, perspective, uneven light, blur."""
    rng = np.random.default_rng(seed)
    img = img.rotate(-3.5, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=235)
    w, h = img.size
    # Perspective: the top edge is slightly narrower than the bottom edge.
    coeffs = _perspective_coeffs(
        [(0, 0), (w, 0), (w, h), (0, h)],
        [(-30, 10), (w + 30, 10), (w, h), (0, h)],
    )
    img = img.transform((w, h), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC, fillcolor=235)
    # Uneven lighting: darker towards the bottom-right corner.
    yy, xx = np.mgrid[0:h, 0:w]
    shade = 1.0 - 0.38 * ((xx / w) * 0.5 + (yy / h) * 0.5)
    pixels = np.asarray(img, dtype=np.float32) * shade
    pixels += rng.normal(0, 6, pixels.shape)
    img = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(1.1))
    return jpeg_roundtrip(img, quality=45)


def _perspective_coeffs(target: list[tuple[float, float]], source: list[tuple[float, float]]) -> list[float]:
    """Solve the 8 coefficients Pillow's PERSPECTIVE transform expects."""
    matrix = []
    for (x, y), (u, v) in zip(target, source):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    a = np.array(matrix, dtype=np.float64)
    b = np.array(source, dtype=np.float64).reshape(8)
    return np.linalg.solve(a, b).tolist()


def jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("L")


# --- the four samples -------------------------------------------------------------


def build() -> dict[str, dict]:
    truth: dict[str, dict] = {}

    lines, total = invoice_lines(
        "INV-88341",
        "14 Sep 2026",
        "14 Oct 2026",
        "PO-55120",
        [("24-port managed switch", 4, 412.00), ("10G SFP+ transceiver", 12, 38.50), ("Cat6 patch cable 2 m", 40, 3.20)],
    )
    render(lines).save(OUT_DIR / "INV-88341_clean.png")
    truth["INV-88341_clean.png"] = {
        "doc_type": "invoice",
        "condition": "clean digital render",
        "text": page_text(lines),
        "fields": {
            "invoice_id": "INV-88341",
            "invoice_date": "2026-09-14",
            "po_number": "PO-55120",
            "vendor_name": "Bluefin Network Supplies Ltd",
            "currency": "EUR",
            "total": total,
        },
    }

    lines, total = invoice_lines(
        "INV-88342",
        "21 Sep 2026",
        "21 Oct 2026",
        "",  # no PO number printed: the field the vision models are tempted to invent
        [("Rack PDU 16 A", 2, 289.00), ("UPS battery module", 3, 164.90), ("Fibre patch lead LC-LC 3 m", 25, 6.80)],
    )
    degrade_scan(render(lines), seed=88342).save(OUT_DIR / "INV-88342_scan.png")
    truth["INV-88342_scan.png"] = {
        "doc_type": "invoice",
        "condition": "low-quality scan: skew, blur, low contrast, speckle",
        "text": page_text(lines),
        "fields": {
            "invoice_id": "INV-88342",
            "invoice_date": "2026-09-21",
            "po_number": None,
            "vendor_name": "Bluefin Network Supplies Ltd",
            "currency": "EUR",
            "total": total,
        },
    }

    ticket = {
        "Incident ID": "INC-104233",
        "Opened": "2026-09-22 08:47 UTC",
        "Reported by": "Finance user (EU)",
        "Priority": "P3",
        "Category": "Endpoint / Disk encryption",
        "Assignment group": "Service Desk L1",
        "Summary": "Laptop asks for disk-encryption recovery key after firmware update",
    }
    description = [
        "After this morning's firmware update the laptop restarts to a",
        "screen asking for a 48-digit recovery key. The user cannot sign in",
        "and has a payroll deadline at 14:00. Asset tag LT-77410.",
    ]
    lines = ticket_lines(ticket, description)
    render(lines).save(OUT_DIR / "INC-104233_ticket.png")
    truth["INC-104233_ticket.png"] = {
        "doc_type": "ticket",
        "condition": "clean printout",
        "text": page_text(lines),
        "fields": {"incident_id": "INC-104233", "priority": "P3", "summary": ticket["Summary"], "kb_hint": "KB-0450"},
    }

    ticket = {
        "Incident ID": "INC-104251",
        "Opened": "2026-09-23 13:05 UTC",
        "Reported by": "Legal user (APAC)",
        "Priority": "P4",
        "Category": "Email / Mailbox",
        "Assignment group": "Service Desk L1",
        "Summary": "Outgoing emails stuck in outbox",
    }
    description = [
        "Messages have stayed in the outbox since yesterday. A mailbox",
        "quota warning shows 98% used. The user must send contract drafts",
        "to the counterparty today.",
    ]
    lines = ticket_lines(ticket, description)
    degrade_photo(render(lines), seed=104251).save(OUT_DIR / "INC-104251_photo.png")
    truth["INC-104251_photo.png"] = {
        "doc_type": "ticket",
        "condition": "phone photo: rotation, perspective, uneven light, blur",
        "text": page_text(lines),
        "fields": {"incident_id": "INC-104251", "priority": "P4", "summary": ticket["Summary"], "kb_hint": "KB-0521"},
    }
    return truth


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Module 1 OCR samples.")
    parser.add_argument("--force", action="store_true", help="regenerate even if the samples exist")
    args = parser.parse_args()
    truth_path = OUT_DIR / "ground_truth.json"
    if truth_path.exists() and not args.force:
        print(f"Samples already exist in {OUT_DIR} (use --force to regenerate).")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    truth = build()
    truth_path.write_text(json.dumps(truth, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, item in truth.items():
        print(f"wrote {name:<24} {item['condition']}")
    print(f"wrote ground_truth.json  ({len(truth)} documents)")


if __name__ == "__main__":
    main()
