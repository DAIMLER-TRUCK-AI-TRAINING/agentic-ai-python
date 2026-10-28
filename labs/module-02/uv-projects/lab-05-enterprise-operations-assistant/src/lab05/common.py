"""Shared set-up for the Module 2 uv projects: .env settings, both SDK clients, LangChain models and print helpers.

The same file sits in each of the five projects, so every project installs and runs on its own.
Labs 01-03 call the two SDKs directly; Labs 04-05 build LangChain models with chat_model(), and
only those two projects install LangChain (it is imported inside chat_model()).
"""

from __future__ import annotations

import functools
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import openai
import pandas as pd
from dotenv import find_dotenv, load_dotenv

ENV_FILE = find_dotenv(usecwd=True)  # walks up from the current folder to the repo-root .env
load_dotenv(ENV_FILE)

PROVIDERS = ("openai", "anthropic")


def find_lab_dir() -> Path:
    """labs/module-02, which holds data/, templates/ and skills/; the project sits inside it."""
    for start in (Path(__file__).resolve(), Path.cwd()):
        for folder in (start, *start.parents):
            if folder.name == "module-02" and (folder / "data").is_dir():
                return folder
    raise SystemExit("❌ labs/module-02/data not found: run the project from inside the course repository.")


LAB_DIR = find_lab_dir()
DATA_DIR = LAB_DIR / "data"
# The notebooks write to outputs/; the uv projects write one level down, so a run never
# overwrites a trace sheet, Skill or review you edited by hand.
OUTPUT_DIR = LAB_DIR / "outputs" / "uv-projects"


def setting(name: str) -> str:
    """A value from .env or the shell, stripped; empty when unset."""
    return os.getenv(name, "").strip()


# Model IDs come from .env, never from code (labs/setup/.env.example has the defaults).
OPENAI_MODEL = setting("OPENAI_MODEL")
OPENAI_SMALL_MODEL = setting("OPENAI_SMALL_MODEL")
ANTHROPIC_MODEL = setting("ANTHROPIC_MODEL")
ANTHROPIC_SMALL_MODEL = setting("ANTHROPIC_SMALL_MODEL")
MODELS = {"openai": OPENAI_MODEL, "anthropic": ANTHROPIC_MODEL}
SMALL_MODELS = {"openai": OPENAI_SMALL_MODEL, "anthropic": ANTHROPIC_SMALL_MODEL}

_selected: set[str] = set(PROVIDERS)
_warned: set[str] = set()


def select_providers(choice: str) -> None:
    """Limit this run to "openai" or "anthropic"; "both" (the default) compares them."""
    _selected.clear()
    _selected.update(PROVIDERS if choice == "both" else (choice,))


@functools.cache
def _openai_or_none() -> openai.OpenAI | None:
    # The SDK reads OPENAI_API_KEY from the environment itself.
    return openai.OpenAI() if setting("OPENAI_API_KEY") and OPENAI_MODEL else None


@functools.cache
def _claude_or_none() -> anthropic.Anthropic | None:
    # The SDK reads ANTHROPIC_API_KEY from the environment itself.
    return anthropic.Anthropic() if setting("ANTHROPIC_API_KEY") and ANTHROPIC_MODEL else None


def openai_client() -> openai.OpenAI:
    """The OpenAI client; call it only after available("openai") or configured("openai")."""
    if (client := _openai_or_none()) is None:
        raise RuntimeError("OpenAI is not configured: set OPENAI_API_KEY and OPENAI_MODEL in .env")
    return client


def claude_client() -> anthropic.Anthropic:
    """The Claude client; call it only after available("anthropic") or configured("anthropic")."""
    if (client := _claude_or_none()) is None:
        raise RuntimeError("Claude is not configured: set ANTHROPIC_API_KEY and ANTHROPIC_MODEL in .env")
    return client


def configured(provider: str) -> bool:
    """True if `provider` is selected for this run and has its key and model."""
    client = _openai_or_none() if provider == "openai" else _claude_or_none()
    return provider in _selected and client is not None


def available(provider: str) -> bool:
    """Like configured(), but say once per run why a selected provider is skipped."""
    if provider in _selected and not configured(provider) and provider not in _warned:
        _warned.add(provider)
        print(f"⚠️  {provider}: not configured, skipping its calls. Add its key and model to .env.")
    return configured(provider)


def active_providers() -> list[str]:
    """The providers this run calls, always in the same order."""
    return [p for p in PROVIDERS if available(p)]


def claude_text(message) -> str:
    """Join the text blocks of a Claude response; thinking blocks are skipped."""
    return "".join(block.text for block in message.content if block.type == "text")


def run_parallel(fn, jobs: list[tuple], max_workers: int = 8) -> list:
    """Call fn(*job) for every job on a small thread pool; results come back in job order."""
    if not jobs:
        return []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(lambda job: fn(*job), jobs))


# --- Labs 04-05: one provider from LLM_PROVIDER, the other for comparison ----------------------


def llm_provider() -> str:
    """LLM_PROVIDER from .env (openai or anthropic; openai when unset)."""
    name = (setting("LLM_PROVIDER") or "openai").lower()
    if name not in PROVIDERS:
        raise SystemExit(f"❌ LLM_PROVIDER must be one of {PROVIDERS}, got {name!r}: fix it in .env.")
    return name


def main_provider() -> str:
    """The provider the lab runs on: the one --provider names, else LLM_PROVIDER."""
    return next(iter(_selected)) if len(_selected) == 1 else llm_provider()


def other_provider() -> str:
    """The comparison provider: the one that is not main_provider()."""
    return "anthropic" if main_provider() == "openai" else "openai"


# Low reasoning effort on both providers; never pass temperature to Claude (current models reject it).
MODEL_KWARGS = {
    "openai": {"reasoning": {"effort": "low"}, "use_responses_api": True},
    "anthropic": {"output_config": {"effort": "low"}, "max_tokens": 8_000},
}


@functools.cache
def chat_model(provider: str, *, small: bool = False):
    """A LangChain chat model for `provider` (its small tier if `small`), or None if it is not usable.

    The same as labs/setup/llm.py's get_chat_model(), which the notebooks import.
    """
    model_id = (SMALL_MODELS if small else MODELS)[provider]
    if not available(provider) or not model_id:
        return None
    from langchain.chat_models import init_chat_model

    return init_chat_model(model_id, model_provider=provider, **MODEL_KWARGS[provider])


# --- Output -------------------------------------------------------------------------------------


def print_status(*, llm_provider_matters: bool = False) -> None:
    """Which .env was loaded and which providers are ready. Key values are never printed."""
    print(f"Lab folder : {LAB_DIR}")
    print(f".env       : {ENV_FILE or 'not found: copy labs/setup/.env.example to the repo root as .env'}")
    print(f"OpenAI     : {'✅ ' + OPENAI_MODEL if _openai_or_none() else '❌ OPENAI_API_KEY or OPENAI_MODEL missing'}")
    print(f"Claude     : {'✅ ' + ANTHROPIC_MODEL if _claude_or_none() else '❌ ANTHROPIC_API_KEY or ANTHROPIC_MODEL missing'}")
    if llm_provider_matters:
        compare = f", compared with {other_provider()}" if len(_selected) == 2 else ""
        print(f"Runs on    : {main_provider()} ({'--provider' if len(_selected) == 1 else 'LLM_PROVIDER'}){compare}")
    elif len(_selected) == 1:
        print(f"Selected   : {next(iter(_selected))} only (--provider)")
    if not (_openai_or_none() or _claude_or_none()):
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
        if pd.api.types.is_numeric_dtype(values):
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
