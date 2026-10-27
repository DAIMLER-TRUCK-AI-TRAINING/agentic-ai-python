#!/usr/bin/env python3
"""Instructor-led demo: cloud OCR with Azure AI Vision Read on the Lab 02 samples.

Instructor only. Participants never need Azure credentials.

    # (instructor) - from the repo root, venv active, AZURE_VISION_* set in .env
    python labs/module-01/instructor/cloud_ocr_demo.py
    python labs/module-01/instructor/cloud_ocr_demo.py --sample INV-88342_scan.png

For each sample image it calls the Image Analysis 4.0 Read feature, prints the
recognised lines with word confidences, and saves the result to
``labs/module-01/data/cloud_ocr/<sample>.azure.json``. Lab 02 Part 4 picks those
files up, if present, and adds the cloud column to the comparison table.

Microsoft has announced that the Image Analysis API (including Read) retires on
25 September 2028 and recommends Azure Document Intelligence (the prebuilt Read
model) for OCR. The outline names Azure AI Vision, so the demo uses it and the
lab shows the migration path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

MODULE_DIR = Path(__file__).resolve().parents[1]
SAMPLES = MODULE_DIR / "data" / "samples"
OUT_DIR = MODULE_DIR / "data" / "cloud_ocr"


def read_image(path: Path) -> dict:
    """Run Azure AI Vision Read on one image and return a JSON-serialisable result."""
    from azure.ai.vision.imageanalysis import ImageAnalysisClient
    from azure.ai.vision.imageanalysis.models import VisualFeatures
    from azure.core.credentials import AzureKeyCredential

    client = ImageAnalysisClient(
        endpoint=os.environ["AZURE_VISION_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["AZURE_VISION_KEY"]),
    )
    started = time.perf_counter()
    result = client.analyze(image_data=path.read_bytes(), visual_features=[VisualFeatures.READ])
    latency = time.perf_counter() - started

    lines = []
    if result.read is not None:
        for block in result.read.blocks:
            for line in block.lines:
                lines.append(
                    {
                        "text": line.text,
                        "words": [{"text": w.text, "confidence": round(w.confidence, 3)} for w in line.words],
                    }
                )
    return {
        "service": "Azure AI Vision - Image Analysis 4.0 Read",
        "model_version": result.model_version,
        "latency_seconds": round(latency, 2),
        "text": "\n".join(line["text"] for line in lines),
        "lines": lines,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Instructor demo: Azure AI Vision Read on the Lab 02 samples.")
    parser.add_argument("--sample", help="one file name from data/samples/ (default: all)")
    args = parser.parse_args()

    load_dotenv(find_dotenv(usecwd=True))
    missing = [v for v in ("AZURE_VISION_ENDPOINT", "AZURE_VISION_KEY") if not os.getenv(v)]
    if missing:
        sys.exit(f"Set {', '.join(missing)} in the repo-root .env first (instructor only).")
    if not (SAMPLES / "ground_truth.json").exists():
        sys.exit("Generate the samples first: python labs/module-01/data/make_samples.py")

    names = [args.sample] if args.sample else sorted(p.name for p in SAMPLES.glob("*.png"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        result = read_image(SAMPLES / name)
        out = OUT_DIR / f"{name}.azure.json"
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        low = [w for line in result["lines"] for w in line["words"] if w["confidence"] < 0.8]
        print(f"\n=== {name}  ({result['latency_seconds']} s, model {result['model_version']})")
        for line in result["lines"]:
            print("  ", line["text"])
        print(f"   {len(low)} word(s) below 0.80 confidence: {[w['text'] for w in low][:10]}")
        print(f"   saved {out.relative_to(MODULE_DIR)}")


if __name__ == "__main__":
    main()
