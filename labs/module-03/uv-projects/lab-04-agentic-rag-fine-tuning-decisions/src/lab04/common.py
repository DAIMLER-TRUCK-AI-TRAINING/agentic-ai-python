"""Shared set-up for the Module 3 uv projects: .env settings, the provider choice, the shared RAG code and print helpers.

The same file sits in each of the four projects, so every project installs and runs on its own.
The models and the RAG helpers are the two files the notebooks import, used in place:
labs/setup/llm.py (the four providers, LangChain and LlamaIndex models, embeddings) and
labs/module-03/rag_kit.py (corpus, connectors, stores, retrieval, access control and references).
"""

from __future__ import annotations

import functools
import logging
import math
import os
import sys
import warnings
from pathlib import Path
from typing import get_args

import pandas as pd
from dotenv import find_dotenv, load_dotenv

ENV_FILE = find_dotenv(usecwd=True)  # walks up from the current folder to the repo-root .env
load_dotenv(ENV_FILE)
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # Chroma: no usage telemetry
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=FutureWarning)
# torch logs a macOS-only notice ("Redirects are currently not supported") when transformers imports it.
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)


def find_lab_dir() -> Path:
    """labs/module-03, which holds data/ and rag_kit.py; the project sits inside it."""
    for start in (Path(__file__).resolve(), Path.cwd()):
        for folder in (start, *start.parents):
            if folder.name == "module-03" and (folder / "rag_kit.py").is_file():
                return folder
    raise SystemExit("❌ labs/module-03/rag_kit.py not found: run the project from inside the course repository.")


LAB_DIR = find_lab_dir()
DATA_DIR = LAB_DIR / "data"
# The notebooks write to outputs/; the uv projects write one level down.
OUTPUT_DIR = LAB_DIR / "outputs" / "uv-projects"
for _path in (str(LAB_DIR.parent / "setup"), str(LAB_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import llm  # noqa: E402  labs/setup/llm.py
import rag_kit as rk  # noqa: E402  labs/module-03/rag_kit.py

# The notebooks keep chroma_db/ and faiss_index/ in labs/module-03. Each uv project keeps its own under
# outputs/uv-projects/lab0N/, so a terminal run never opens a store that a notebook kernel or another
# project has open (Chroma's on-disk client is not meant to be shared between processes).
STORE_DIR = OUTPUT_DIR / Path(__file__).parent.name
rk.CHROMA_DIR = STORE_DIR / "chroma_db"
rk.FAISS_DIR = STORE_DIR / "faiss_index"

PROVIDERS: tuple[str, ...] = get_args(llm.Provider)

_only: str | None = None  # --provider
_warned: set[str] = set()


def select_provider(choice: str | None) -> None:
    """--provider: run on that provider only, with no comparison. None: LLM_PROVIDER plus its partner."""
    global _only
    _only = choice


def main_provider() -> str:
    """The provider the lab runs on: the one --provider names, else LLM_PROVIDER (openai when unset)."""
    try:
        return llm.get_provider(_only)
    except llm.ProviderConfigError as exc:
        raise SystemExit(f"❌ {exc}: fix it in .env.") from None


def needs_model(what: str) -> bool:
    """True if the main provider is configured; otherwise say that `what` is skipped."""
    if llm.is_configured(main_provider()):
        return True
    print(f"⚠️  {main_provider()} is not configured: skipping {what}. Add its key and model to .env.")
    return False


def comparison_provider() -> str | None:
    """The other API family on the same platform, for the comparison steps; None with --provider,
    or when it is not configured (said once per run)."""
    if _only:
        return None
    partner = llm.partner_provider(main_provider())
    if llm.is_configured(partner):
        return partner
    if partner not in _warned:
        _warned.add(partner)
        print(f"⚠️  {partner} is not configured: skipping the comparison. Add its key and model to .env.")
    return None


def providers() -> list[str]:
    """The providers this run calls: the main one, then the comparison one when there is one."""
    if not llm.is_configured(main_provider()):
        return []
    return [p for p in (main_provider(), comparison_provider()) if p]


@functools.cache
def setup_llama() -> None:
    """LlamaIndex's global embedding model, and its LLM when the main provider is configured.

    Without a provider, Settings.llm is a MockLLM, so the local steps still run: chunking,
    indexing, BM25, fusion with num_queries=1 and re-ranking never call the LLM.
    """
    from llama_index.core import Settings
    from llama_index.core.llms import MockLLM

    if llm.is_configured(main_provider()):
        rk.setup_llama(main_provider())
    else:
        Settings.embed_model = llm.get_llama_embed_model()
        Settings.llm = MockLLM()


# --- Checks -------------------------------------------------------------------------------------

FAILED: list[str] = []


def check(name: str, fn) -> None:
    """Run a check function, as the notebooks' check cells do: ✅ passed, ❌ failed, ⚠️ skipped
    (the function returned the reason). The command exits with status 1 if any check failed."""
    try:
        skipped = fn()
    except AssertionError as exc:
        FAILED.append(name)
        print(f"❌ {name}" + (f": {exc}" if str(exc) else ""))
        return
    print(f"⚠️  {name}: skipped ({skipped})" if skipped else f"✅ {name}")


# --- Output -------------------------------------------------------------------------------------


def _model_state(provider: str) -> str:
    if not llm.is_configured(provider):
        return "❌ key, model or Azure resource missing in .env"
    return f"✅ {llm.get_model_name(provider)}"


def print_status() -> None:
    """Which .env was loaded, the providers, the embeddings and the stores. Key values are never printed."""
    main = main_provider()
    partner = llm.partner_provider(main)
    try:
        embeddings = f"{llm.get_embedding_provider()} · {llm.get_embedding_model_name()} (index suffix {llm.embedding_slug()})"
    except llm.ProviderConfigError as exc:
        embeddings = f"❌ {exc}"
    print(f"Lab folder : {LAB_DIR}")
    print(f".env       : {ENV_FILE or 'not found: copy labs/setup/.env.example to the repo root as .env'}")
    print(f"Runs on    : {main} ({'--provider' if _only else 'LLM_PROVIDER'}) {_model_state(main)}")
    print(f"Compared   : {'no (--provider)' if _only else partner + ' ' + _model_state(partner)}")
    print(f"Embeddings : {embeddings}")
    print(f"Stores     : {rk.CHROMA_DIR.relative_to(LAB_DIR)}/ · Pinecone "
          f"{'✅ PINECONE_API_KEY present' if rk.vector_backend() == 'pinecone' else 'not configured (Chroma is used)'}")
    if not llm.is_configured(main):
        print("ℹ️  No provider is configured: steps that call a model are skipped; the local steps still run.")


def heading(title: str) -> None:
    print(f"\n{title}\n{'─' * len(title)}")


def step(text: str) -> None:
    print(f"\n▸ {text}")


def show(frame: pd.DataFrame, *, index: bool = True, max_width: int = 90) -> None:
    """Print a DataFrame in full, text left-aligned and cut at `max_width`, or say that it is empty."""
    if frame.empty:
        print("(no rows)")
        return
    def clip(value) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "—"
        text = " ".join(str(value).split())  # one line per row, even for a multi-line model reply
        return text if len(text) <= max_width else text[: max_width - 1] + "…"

    frame = frame.copy()
    formatters = {}
    for column in frame.columns:
        values = frame[column]
        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            if not values.isna().any():
                continue  # numbers stay right-aligned
            # A gap in a numeric column (a row the metric does not apply to) prints as —, not NaN.
            whole = values.dropna().mod(1).eq(0).all()
            values = [v if pd.isna(v) else int(v) if whole else v for v in values]
        cells = [clip(v) for v in values]
        width = max(len(c) for c in cells)
        frame[column] = cells
        formatters[column] = lambda value, width=width: value.ljust(width)
    print(frame.to_string(index=index, formatters=formatters, justify="left"))
