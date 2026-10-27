"""Part 2 — temperature on a local model you can inspect, then what the hosted APIs allow."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence

# PyTorch logs a harmless note on macOS when transformers imports torch.distributed; keep the demo output clean.
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

import anthropic  # noqa: E402
import openai  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402

from .common import (
    ANTHROPIC_MODEL,
    OPENAI_MODEL,
    OPENAI_SMALL_MODEL,
    active_providers,
    available,
    claude_client,
    openai_client,
    show,
    step,
)
from .part1_first_calls import ASK

LOCAL_LM = "HuggingFaceTB/SmolLM2-135M"
NEXT_TOKEN_PROMPT = "If you forget your password, click the link to"
TEMPERATURES = (0.2, 0.7, 1.0, 1.5)

# Used only when SmolLM2 is neither cached nor downloadable: ILLUSTRATIVE numbers, not a real model.
FALLBACK_TOKENS = [" reset", " the", " log", " change", " get", " your"]
FALLBACK_LOGITS = (4.1, 3.6, 1.6, 0.0, 0.0, -0.5)

SWEEP_PROMPT = "Write a one-line status-page headline for a 20-minute VPN outage that is now resolved."
TASKS = {
    "constrained": (
        "Classify the priority of this ticket as exactly one of P1, P2, P3, P4. Reply with the label only.\n"
        "Ticket: The whole EU-West office has no network access and payroll cannot run."
    ),
    "open": SWEEP_PROMPT,
}


def load_local_lm():
    """(tokenizer, model) for SmolLM2-135M, or (None, None) with a note when it cannot be loaded."""
    hf_logging.disable_progress_bar()
    try:
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_LM)
        model = AutoModelForCausalLM.from_pretrained(LOCAL_LM).eval()
    except OSError as exc:
        print(f"⚠️  Could not load {LOCAL_LM} ({exc.__class__.__name__}). Run labs/setup/prefetch_models.py.")
        print("   Until then this part uses an ILLUSTRATIVE logits vector, not a real model.")
        return None, None
    print(f"Loaded {LOCAL_LM}: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters")
    return tokenizer, model


def next_token_logits(tokenizer, model) -> tuple[torch.Tensor, Callable[[int], str]]:
    """One score per vocabulary entry for the token after NEXT_TOKEN_PROMPT, and a label for any index."""
    if model is None:
        return torch.tensor(FALLBACK_LOGITS), lambda index: repr(FALLBACK_TOKENS[index])
    with torch.no_grad():
        inputs = tokenizer(NEXT_TOKEN_PROMPT, return_tensors="pt")
        logits = model(**inputs).logits[0, -1]
    return logits, lambda index: repr(tokenizer.decode(index))


def entropy(probs: torch.Tensor) -> float:
    """Shannon entropy in nats: 0 means certain, higher means more spread out."""
    return float(-(probs * torch.log(probs + 1e-12)).sum())


def next_token_table(
    logits: torch.Tensor, temperatures: Sequence[float], label: Callable[[int], str], k: int = 6
) -> pd.DataFrame:
    """Probability of the k most likely next tokens under each temperature."""
    top_ids = torch.topk(logits, k).indices
    columns = {f"T={t}": torch.softmax(logits / t, dim=-1)[top_ids].tolist() for t in temperatures}
    return pd.DataFrame(columns, index=[label(int(i)) for i in top_ids]).round(3)


def sample_continuations(tokenizer, model, temperature: float | None, n: int = 20, seed: int = 7) -> list[str]:
    """Generate n continuations of NEXT_TOKEN_PROMPT; temperature=None means greedy decoding."""
    torch.manual_seed(seed)
    inputs = tokenizer(NEXT_TOKEN_PROMPT, return_tensors="pt")
    if temperature is None:
        sampling = {"do_sample": False}
    else:  # top_k=0 and top_p=1.0 switch off the other filters, so only temperature acts
        sampling = {"do_sample": True, "temperature": temperature, "top_k": 0, "top_p": 1.0}
    outputs = model.generate(
        **inputs,
        max_new_tokens=8,
        num_return_sequences=1 if temperature is None else n,
        pad_token_id=tokenizer.eos_token_id,
        **sampling,
    )
    start = inputs["input_ids"].shape[1]
    return [tokenizer.decode(o[start:], skip_special_tokens=True) for o in outputs]


def local_temperature() -> None:
    step(f"Temperature on a local model: {LOCAL_LM}")
    tokenizer, model = load_local_lm()
    logits, label = next_token_logits(tokenizer, model)
    print(f"Prompt: {NEXT_TOKEN_PROMPT!r} → {logits.shape[0]:,} candidate next tokens\n")
    for t in TEMPERATURES:
        probs = torch.softmax(logits / t, dim=-1)
        print(f"T={t}: entropy {entropy(probs):5.2f} nats · top-1 probability {float(probs.max()):.3f}")
    print()
    show(next_token_table(logits, TEMPERATURES, label))

    if model is not None:
        step("20 sampled continuations per temperature: how many are distinct?")
        for t in (None, 0.3, 1.0, 1.5):
            name = "greedy" if t is None else f"T={t}"
            continuations = sample_continuations(tokenizer, model, t)
            print(f"{name:>7}: {len(set(continuations)):>2} distinct of {len(continuations):>2} · e.g. {continuations[-1]!r}")


def temperature_rejections() -> None:
    """The deliberate failure of this lab: send temperature where the API no longer accepts it."""
    if available("openai"):
        step(f"Deliberate failure 1: temperature on {OPENAI_MODEL} at its default reasoning effort")
        try:
            openai_client().responses.create(
                model=OPENAI_MODEL, input="Say hello.", temperature=0.2, max_output_tokens=200
            )
            print(f"{OPENAI_MODEL} accepted temperature at its default reasoning effort.")
        except openai.BadRequestError as exc:
            print(f"❌ {OPENAI_MODEL} → 400 BadRequestError: {exc.message}")

    if available("anthropic"):
        step(f"Deliberate failure 2: temperature on {ANTHROPIC_MODEL}, through the SDK, then forced onto the wire")
        hello = [{"role": "user", "content": "Say hello."}]
        try:
            claude_client().messages.create(model=ANTHROPIC_MODEL, max_tokens=200, messages=hello, temperature=0.2)
        except TypeError as exc:
            print(f"❌ The SDK refuses before sending anything: TypeError: {exc}")
        try:
            claude_client().messages.create(
                model=ANTHROPIC_MODEL, max_tokens=200, messages=hello, extra_body={"temperature": 0.2}
            )
            print(f"{ANTHROPIC_MODEL} accepted temperature=0.2.")
        except anthropic.BadRequestError as exc:
            print(f"❌ {ANTHROPIC_MODEL} → 400 BadRequestError: {exc.message}")


def first_token_alternatives(response, k: int = 5) -> list[tuple[str, float]]:
    """The k most likely first tokens and their probabilities, from the response logprobs."""
    for item in response.output:
        if item.type == "message":
            for part in item.content:
                if part.type == "output_text" and part.logprobs:
                    return [(alt.token, round(math.exp(alt.logprob), 3)) for alt in part.logprobs[0].top_logprobs[:k]]
    return []


def temperature_sweep() -> None:
    """Where temperature still works: OPENAI_SMALL_MODEL with reasoning effort "none"."""
    if not available("openai"):
        return
    if not OPENAI_SMALL_MODEL:
        print("⚠️  OPENAI_SMALL_MODEL is not set: skipping the temperature sweep.")
        return
    step(f"Temperature sweep on {OPENAI_SMALL_MODEL} (reasoning effort 'none'), 4 runs each")
    try:
        for t in (0.0, 1.0, 1.8):
            responses = [
                openai_client().responses.create(
                    model=OPENAI_SMALL_MODEL,
                    input=SWEEP_PROMPT,
                    reasoning={"effort": "none"},
                    temperature=t,
                    max_output_tokens=60,
                    include=["message.output_text.logprobs"],
                    top_logprobs=5,
                )
                for _ in range(4)
            ]
            texts = [response.output_text.strip() for response in responses]
            alternatives = first_token_alternatives(responses[-1])
            print(f"T={t}: {len(set(texts))} distinct of {len(texts)} · first-token alternatives {alternatives}")
            for text in texts:
                print("     ", text)
    except openai.BadRequestError as exc:
        print(f"❌ {OPENAI_SMALL_MODEL} rejected the sweep: {exc.message}")
        print("   OPENAI_SMALL_MODEL must support reasoning effort 'none' (see labs/setup/.env.example).")


def default_sampling_variability(runs: int = 3) -> None:
    """Default sampling on both main models: a constrained task versus an open one."""
    providers = active_providers()
    if not providers:
        return
    step(f"Default sampling: a constrained task versus an open one, {runs} runs each")
    rows = []
    for provider in providers:
        for task, prompt in TASKS.items():
            outputs = [ASK[provider](prompt) for _ in range(runs)]
            rows.append(
                {"provider": provider, "task": task, "distinct": len(set(outputs)), "runs": runs, "first output": outputs[0][:90]}
            )
    show(pd.DataFrame(rows), index=False)


def run() -> None:
    local_temperature()
    temperature_rejections()
    temperature_sweep()
    default_sampling_variability()
