# Module 1 — instructor materials

> **Instructor only.** Participants never need the credentials this folder uses.

| File | Used in | What it does |
| ---- | ------- | ------------ |
| [`cloud_ocr_demo.py`](cloud_ocr_demo.py) | Lab 02 Part 4 (5 min) | Runs Azure AI Vision Read on the four Lab 02 samples, prints lines and word confidences, and saves `data/cloud_ocr/<sample>.azure.json` |

## Before the session

1. Get the cloud sandbox details from the client IT team. Lab infrastructure is
   still to be finalised with them (course map §4 #6), so do not assume a
   tenant, an endpoint or a key.
2. Put the endpoint and key in **your** repo-root `.env` only:
   `AZURE_VISION_ENDPOINT=` and `AZURE_VISION_KEY=`. Never share your screen
   while `.env` is open.
3. Generate the samples and do a dry run:

   ```bash
   # (instructor) - from the repo root, venv active
   python labs/module-01/data/make_samples.py
   python labs/module-01/instructor/cloud_ocr_demo.py
   ```

## During Lab 02 Part 4

- Run the script live on `INV-88342_scan.png` and `INC-104251_photo.png`.
  Point out the reading order of the table rows, the word confidences, and the
  latency of a network round trip.
- If you want learners to run the comparison cell themselves, share the four
  `data/cloud_ocr/*.azure.json` files (they contain only the OCR text of
  synthetic documents). Learners copy them into their own `data/cloud_ocr/`.
- Close with the currency note. Microsoft retires the Image Analysis API,
  Read included, on **25 September 2028** and recommends **Azure Document
  Intelligence** (prebuilt Read model) for OCR. AWS Textract is the AWS
  counterpart. The outline covers it as an overview only, with no lab.

## If the cloud sandbox is not available

Say so, show the observation table in Lab 02 Part 4, and talk through the
trade-offs: residency, cost per page, layout output, confidence scores and
offline capability. Learners still compare Tesseract with AI-assisted OCR from
OpenAI and Claude, which needs only the course API keys.
