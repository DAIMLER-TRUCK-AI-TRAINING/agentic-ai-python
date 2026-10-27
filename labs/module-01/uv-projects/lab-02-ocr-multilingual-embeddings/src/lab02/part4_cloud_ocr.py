"""Part 4 — cloud OCR comparison: add the instructor's Azure AI Vision Read results to the CER table."""

from __future__ import annotations

import json

from .common import DATA_DIR, show, step
from .part1_tesseract import ground_truth, tesseract_ok
from .part2_accuracy import cer, cer_table

CLOUD_DIR = DATA_DIR / "cloud_ocr"


def run() -> None:
    files = sorted(CLOUD_DIR.glob("*.azure.json")) if CLOUD_DIR.is_dir() else []
    if not files:
        print(
            "ℹ️  No cloud OCR results in data/cloud_ocr/. The instructor creates them with\n"
            "   labs/module-01/instructor/cloud_ocr_demo.py (course venv, Azure credentials); learners need none.\n"
            "   Watch the demo and fill in the observation table in the lab guide."
        )
        return
    step("CER with the instructor's Azure AI Vision Read results added")
    truth = ground_truth()
    azure = {}
    for path in files:
        name = path.name.removesuffix(".azure.json")
        if name in truth:
            azure[name] = cer(json.loads(path.read_text(encoding="utf-8"))["text"], truth[name]["text"])
    table = cer_table().copy() if tesseract_ok() else None
    if table is None:
        print("(Tesseract is not installed: showing the cloud column only)")
        for name, value in azure.items():
            print(f"{name:<24} {value:.3f}")
        return
    table["Azure AI Vision Read"] = table.index.map(azure)
    show(table.round(3))
    print(
        "\nCurrency note: Microsoft retires the Image Analysis API, Read included, on 25 September 2028 and\n"
        "recommends Azure Document Intelligence (prebuilt Read model) for OCR."
    )
