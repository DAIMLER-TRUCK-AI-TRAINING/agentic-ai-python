# Demo 05 — Vision / Multimodal Understanding

The other half of multimodal AI. Demo 04 went **text → image**; this goes **image → text**. A multimodal chat model reads an image *alongside* a text question and answers in natural language.

> Companion to **Guide 06 — Retrieval & Multimodal AI**, §7 (Introduction to Multimodal AI).

---

## Features

* **Image + text in one message** — the model reasons over both
* **URL or local file** — http(s) URLs pass through; local files are base64-encoded into a data URL
* **Practical task** — describe the scene and flag safety concerns

---

## Setup

```bash
uv sync
```

Rename `.envbackup` to `.env` and add your key:

```bash
OPENAI_API_KEY=your_key_here
VISION_MODEL=gpt-4o-mini          # optional
# IMAGE_SOURCE=./my_photo.jpg     # optional: URL or local path
```

---

## Usage

```bash
uv run main.py
```

Works out of the box against a public sample image. To analyze your own, set `IMAGE_SOURCE` in `.env` to a URL or a local file path.

---

## Example Output

```text
Vision model: gpt-4o-mini
Image: https://upload.wikimedia.org/.../Gfp-wisconsin-madison-the-nature-boardwalk.jpg
Question: Describe this image in two sentences, then list any safety concerns.

Analyzing image...

=== Answer ===

A wooden boardwalk leads through tall green grass under a bright blue sky with
scattered clouds. Safety concerns: the narrow path has no railings, and wet wood
can be slippery after rain.
```

---

## Key Concepts

| Concept | Description |
| ------- | ----------- |
| Multimodal message | `content` is a list mixing `text` and `image_url` parts |
| Data URL | Local files become `data:image/...;base64,...` |
| Vision use cases | Document extraction, chart reading, alt-text, visual QA, defect detection |

---

## Try This

* Point `IMAGE_SOURCE` at a receipt and ask the model to extract line items as JSON.
* Ask for **alt-text** suitable for screen readers.
* Give it a chart screenshot and ask for the underlying numbers.
