"""Part 5 — multilingual embeddings: an English-only model against a multilingual one."""

from __future__ import annotations

import functools
import logging
import re

# PyTorch logs a harmless note on macOS when transformers imports torch.distributed; keep the demo output clean.
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

import pandas as pd  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402

from .common import show, step  # noqa: E402

EMBEDDING_MODELS = {
    "english-only": "sentence-transformers/all-MiniLM-L6-v2",
    "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}
SAME_SENTENCE = {
    "en": "The VPN client rejects my new password",
    "de": "Der VPN-Client lehnt mein neues Passwort ab",
    "fr": "Le client VPN refuse mon nouveau mot de passe",
    "es": "El cliente VPN rechaza mi nueva contraseña",
    "hi": "VPN क्लाइंट मेरा नया पासवर्ड अस्वीकार कर रहा है",
    "ja": "VPNクライアントが新しいパスワードを拒否します",
}
# Meaning versus shared words, and two traps: negation and identifiers (guide §7.5).
PAIRS = [
    ("My laptop won't connect to the VPN", "Remote-access tunnel fails to establish"),  # same meaning, no shared words
    ("My laptop won't connect to the VPN", "My laptop won't connect to the printer"),  # shared words, different problem
    ("My laptop won't connect to the VPN", "The canteen is closed today"),  # unrelated
    ("The VPN works after the password change", "The VPN does not work after the password change"),  # negation
    ("Escalate INC-104233 to Network Ops", "Escalate INC-104234 to Network Ops"),  # different incident
]


@functools.cache
def load_encoders() -> dict[str, SentenceTransformer]:
    """Both Sentence Transformers models that are available (cached by labs/setup/prefetch_models.py)."""
    hf_logging.disable_progress_bar()
    encoders = {}
    for label, model_id in EMBEDDING_MODELS.items():
        try:
            encoders[label] = SentenceTransformer(model_id)
        except OSError as exc:
            print(f"⚠️  {model_id} unavailable ({exc.__class__.__name__}): run labs/setup/prefetch_models.py")
        else:
            print(f"{label:>12}: {model_id} · {encoders[label].get_embedding_dimension()} dimensions")
    return encoders


def cross_lingual(encoders: dict[str, SentenceTransformer]) -> pd.DataFrame:
    """Cosine similarity of each language's version of the sentence to the English one, per model."""
    scores = {}
    for label, encoder in encoders.items():
        vectors = encoder.encode(list(SAME_SENTENCE.values()), normalize_embeddings=True)
        scores[label] = dict(zip(SAME_SENTENCE, (vectors @ vectors[0]).round(2)))
    return pd.DataFrame(scores).T


def word_overlap(a: str, b: str) -> float:
    """Jaccard overlap of lower-cased words: a lexical baseline."""
    wa, wb = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return len(wa & wb) / len(wa | wb)


def run() -> None:
    step("Loading the embedding models")
    encoders = load_encoders()
    if not encoders:
        print("ℹ️  No embedding model available: skipping this part.")
        return

    step("One sentence, six languages: similarity of each version to the English one")
    show(cross_lingual(encoders))

    if "multilingual" in encoders:
        step("The full 6 × 6 similarity matrix, multilingual model")
        vectors = encoders["multilingual"].encode(list(SAME_SENTENCE.values()), normalize_embeddings=True)
        show(pd.DataFrame(vectors @ vectors.T, index=list(SAME_SENTENCE), columns=list(SAME_SENTENCE)).round(2))

    step("Meaning versus shared words, and the negation and identifier traps")
    rows = []
    for a, b in PAIRS:
        row = {"sentence A": a, "sentence B": b, "word overlap": round(word_overlap(a, b), 2)}
        for label, encoder in encoders.items():
            va, vb = encoder.encode([a, b], normalize_embeddings=True)
            row[f"cosine ({label})"] = round(float(va @ vb), 2)
        rows.append(row)
    show(pd.DataFrame(rows), index=False, max_width=48)
