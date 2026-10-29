# Demo 04 — Text-to-Image Generation

Generate an image from a text prompt with the **OpenAI Images API** (`gpt-image-1`), and practice the prompt-engineering habit from the guide: build a prompt from **explicit visual dimensions** (subject, setting, composition, lighting, color, style, detail).

> Companion to **Guide 06 — Retrieval & Multimodal AI**, §8 (Diffusion / Text-to-Image) and §9 (Image Prompting).

> ⚠️ **This demo calls a paid image endpoint.** Each run generates one image and incurs a small cost.

---

## Features

* **Images API** — `client.images.generate(...)` with `gpt-image-1`
* **Structured prompt builder** — one Python list, one dimension per line
* **Base64 → PNG** — decodes the response and writes a timestamped file
* **Size / quality controls** — square/landscape/portrait, low/medium/high

---

## Setup

```bash
uv sync
```

Rename `.envbackup` to `.env` and add your key:

```bash
OPENAI_API_KEY=your_key_here
IMAGE_MODEL=gpt-image-1   # optional
```

---

## Usage

```bash
uv run main.py
```

A file like `output_20260616_143012.png` appears next to the script (git-ignored).

---

## Example Output

```text
Prompt:
  A red fox curled asleep, in a snowy pine forest, wide establishing shot,
  soft golden-hour backlight, cool blue shadows, watercolor illustration,
  delicate paper texture, highly detailed

Generating image...
Saved -> output_20260616_143012.png  (1480 KB)
```

---

## The Image-Prompt Anatomy

| Dimension | Example used here |
| --------- | ----------------- |
| Subject + pose | "A red fox curled asleep" |
| Setting | "in a snowy pine forest" |
| Composition | "wide establishing shot" |
| Lighting | "soft golden-hour backlight" |
| Color / mood | "cool blue shadows" |
| Style / medium | "watercolor illustration" |
| Detail cues | "delicate paper texture, highly detailed" |

---

## Try This (change ONE variable at a time)

* **Style:** swap `watercolor illustration` → `35mm film photo` or `isometric 3D render`.
* **Lighting:** `golden-hour backlight` → `dramatic rim light` or `soft studio light`.
* **Size:** pass `size="1024x1536"` for a portrait composition.
* **Quality:** `quality="high"` for finer detail (slower, costlier).

Changing one dimension per run is how you learn what each control actually does.
