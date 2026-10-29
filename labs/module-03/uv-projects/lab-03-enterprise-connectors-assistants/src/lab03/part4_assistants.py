"""Part 4 — one retrieval-and-answer pattern, two assistants: internal policy search and fleet-customer support."""

from __future__ import annotations

import functools

import pandas as pd
from llama_index.core import PromptTemplate
from llama_index.core.postprocessor import SimilarityPostprocessor

from .common import check, main_provider, needs_model, rk, show, step
from .part3_access import access_filters, knowledge_base

POLICY_SEARCH = rk.RagProfile(
    "policy-search", ("policy", "contract"),
    "You answer Daimler Truck employees' and dealers' questions about warranty, goodwill, software and data "
    "policies and service contracts. Cite the document id and section, for example [POL-WAR-001 §3]. "
    "If two documents disagree, show both.",
)
CUSTOMER_SUPPORT = rk.RagProfile(
    "customer-support", ("faq",),
    "You help Daimler Truck fleet customers and drivers. Use plain, friendly language. Never mention internal "
    "teams, bulletins or systems that are not in the sources. Cite [KB-…] ids.",
)
# assistant -> (profile, the authenticated caller, [(question, answerable)])
CASES = {
    "policy-search": (POLICY_SEARCH, rk.EU_AFTERSALES, [
        ("How long is the HV battery warranty on an ED-26, and what capacity is guaranteed?", True),
        ("A customer asks for €4,000 goodwill on a gearbox 8 months after warranty. Who approves it?", True),
        ("What uptime does contract CTR-2026-118 guarantee, and what are the service credits?", True),
        ("What is the warranty on aftermarket trailers?", False),
    ]),
    "customer-support": (CUSTOMER_SUPPORT, rk.FLEET_CUSTOMER, [
        ("How long does a DC charge from 20 to 80 percent take?", True),
        ("The AdBlue lamp is flashing. Can I keep driving?", True),
        ("My truck's software update is stuck. What should I do?", True),
        ("Can I get a discount on my next truck?", False),
    ]),
}


def make_assistant(profile: rk.RagProfile, user: rk.User, index, llm=None):
    """A query engine that: retrieves 2 * profile.k candidates with access_filters(user, profile.doc_types),
    re-ranks them to profile.k with rk.reranker(profile.k), drops nodes scoring below profile.floor
    (SimilarityPostprocessor), and answers with profile.instructions + rk.ANSWER_RULES in the prompt."""
    prompt = PromptTemplate(
        f"{profile.instructions}\n{rk.ANSWER_RULES}\n\n<sources>\n{{context_str}}\n</sources>\n\nQuestion: {{query_str}}\nAnswer: "
    )
    return index.as_query_engine(
        llm=llm,
        similarity_top_k=2 * profile.k,
        filters=access_filters(user, list(profile.doc_types)),
        node_postprocessors=[rk.reranker(profile.k), SimilarityPostprocessor(similarity_cutoff=profile.floor)],
        text_qa_template=prompt,
    )


@functools.cache
def run_cases(provider: str) -> pd.DataFrame:
    """Every case through both assistants on `provider`; once per process, so Part 5 reuses it."""
    index, _, _ = knowledge_base()
    llm = rk.llama_llm(provider)
    rows = []
    for name, (profile, user, questions) in CASES.items():
        assistant = make_assistant(profile, user, index, llm)
        for question, answerable in questions:
            with rk.Timer() as t:
                response = assistant.query(question)
            sources = set(rk.doc_ids(response.source_nodes))
            rows.append({"provider": provider, "assistant": name, "question": question[:60], "answerable": answerable,
                         "answered": rk.answered(response), "cited": sorted(rk.cited_ids(str(response))),
                         "citations valid": rk.cited_ids(str(response)) <= sources, "s": t.seconds, "answer": str(response)})
    return pd.DataFrame(rows)


def run() -> None:
    if not needs_model("the two assistants"):
        return
    step(f"Real questions for each assistant, including ones they must not answer ({main_provider()})")
    results = run_cases(main_provider())
    show(results[["assistant", "question", "answerable", "answered", "cited", "citations valid", "s"]], index=False, max_width=60)

    step("Read two answers: one policy answer and one customer answer")
    for i in (0, 6):
        print(f"--- {results.loc[i, 'assistant']}: {results.loc[i, 'question']}\n{results.loc[i, 'answer']}\n")

    def check_part4():
        for _, row in results.iterrows():
            assert row["answered"] == row["answerable"], (row["question"], row["answered"])
            assert row["citations valid"], row["question"]
        support = " ".join(results.loc[results["assistant"] == "customer-support", "answer"])
        assert "TSB-" not in support and "After-Sales Technical Support" not in support

    check("Part 4 — two assistants from one pattern", check_part4)
