"""Part 6 — match multilingual tickets to KB articles, then run the image → OCR → embedding → KB pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import DATA_DIR, available, openai_client, setting, show, step
from .part1_tesseract import TICKET_IMAGES, ground_truth, open_sample, tesseract_ok
from .part2_accuracy import ocr, preprocess
from .part5_similarity import load_encoders

MATCH_COLUMNS = ["ticket_id", "language", "gold_kb", "predicted_kb", "score", "correct"]
ESCALATE_BELOW = 0.35  # a best score under this means "no confident match": route to a human


def load_tickets() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "multilingual_tickets.csv")


def load_kb() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "kb_articles.csv")


def kb_texts(kb: pd.DataFrame) -> list[str]:
    return (kb.title + ". " + kb.summary).tolist()


def match_tickets(encoder, tickets: pd.DataFrame, kb: pd.DataFrame) -> pd.DataFrame:
    """One row per ticket with its best-matching KB article, its cosine score and whether it is right.

    `encoder` is anything with SentenceTransformer's encode(texts, normalize_embeddings=True).
    """
    kb_vectors = encoder.encode(kb_texts(kb), normalize_embeddings=True)
    ticket_vectors = encoder.encode(tickets.text.tolist(), normalize_embeddings=True)
    similarity = ticket_vectors @ kb_vectors.T  # cosine, because the vectors are normalised
    best = similarity.argmax(axis=1)
    result = tickets[["ticket_id", "language", "gold_kb"]].copy()
    result["predicted_kb"] = kb.kb_id.to_numpy()[best]
    result["score"] = similarity.max(axis=1).round(3)
    result["correct"] = result.predicted_kb == result.gold_kb
    return result[MATCH_COLUMNS]


def ticket_text(name: str) -> str:
    """OCR text of a ticket image; falls back to the ground-truth text without Tesseract."""
    if not tesseract_ok():
        return ground_truth()[name]["text"]
    cleaned, _ = preprocess(open_sample(name))
    return ocr(cleaned, 4)


def hosted_embeddings(tickets: pd.DataFrame, kb: pd.DataFrame) -> None:
    """Optional: OpenAI's hosted embedding model on the same task.

    Claude has no embedding endpoint of its own; Anthropic points to Voyage AI for embeddings.
    """
    model = setting("OPENAI_EMBEDDING_MODEL")
    if not model:
        print("ℹ️  OPENAI_EMBEDDING_MODEL is not set: skipping the hosted comparison.")
        return
    if not available("openai"):
        return

    def embed(texts: list[str]) -> np.ndarray:
        data = openai_client().embeddings.create(model=model, input=texts).data
        vectors = np.array([item.embedding for item in data])
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    step(f"Optional: OpenAI's hosted embeddings ({model}) on the same tickets")
    similarity = embed(tickets.text.tolist()) @ embed(kb_texts(kb)).T
    correct = kb.kb_id.to_numpy()[similarity.argmax(axis=1)] == tickets.gold_kb.to_numpy()
    print(f"{model}: top-1 accuracy {correct.mean():.2f} on {len(tickets)} tickets")


def run() -> None:
    tickets, kb = load_tickets(), load_kb()
    step(f"{len(tickets)} tickets {tickets.language.value_counts().to_dict()} · {len(kb)} KB articles")
    show(kb[["kb_id", "title", "owner_team"]], index=False)

    encoders = load_encoders()
    if not encoders:
        print("ℹ️  No embedding model available: skipping the matching.")
        return
    matches = {label: match_tickets(encoder, tickets, kb) for label, encoder in encoders.items()}

    step("Top-1 accuracy per language")
    per_language = pd.DataFrame({label: m.groupby("language").correct.mean() for label, m in matches.items()})
    per_language.loc["ALL"] = [m.correct.mean() for m in matches.values()]
    show(per_language.round(2))
    print({label: f"median best score {m.score.median():.2f}" for label, m in matches.items()})

    step(f"Production habit: a best score under {ESCALATE_BELOW} goes to a human")
    for label, m in matches.items():
        print(f"{label:>12}: {int((m.score < ESCALATE_BELOW).sum()):>2} of {len(m)} tickets would be escalated")

    if "multilingual" in encoders:
        step("End to end: ticket IMAGE → OCR → multilingual embedding → KB article")
        if not tesseract_ok():
            print("(Tesseract is not installed: using the ground-truth text instead of OCR)")
        encoder = encoders["multilingual"]
        kb_vectors = encoder.encode(kb_texts(kb), normalize_embeddings=True)
        pipeline = []
        for name in TICKET_IMAGES:
            scores = kb_vectors @ encoder.encode([ticket_text(name)], normalize_embeddings=True)[0]
            best = int(scores.argmax())
            pipeline.append(
                {
                    "image": name,
                    "predicted": kb.kb_id[best],
                    "title": kb.title[best],
                    "score": round(float(scores[best]), 3),
                    "expected": ground_truth()[name]["fields"]["kb_hint"],
                }
            )
        show(pd.DataFrame(pipeline), index=False)

    hosted_embeddings(tickets, kb)
