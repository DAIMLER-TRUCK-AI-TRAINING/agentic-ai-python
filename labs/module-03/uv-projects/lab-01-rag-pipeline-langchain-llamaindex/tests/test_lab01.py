"""Offline tests: the deterministic parts of Lab 01. No API key, model download or network needed."""

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from llama_index.core.node_parser import SentenceSplitter

from lab01 import part2_faiss
from lab01.common import rk
from lab01.part1_corpus import records
from lab01.part2_faiss import chunks, retrieve
from lab01.part3_chain import format_sources


def test_corpus_has_twenty_documents_with_one_schema():
    corpus = records()
    assert len(corpus) == 20
    assert {r["metadata"]["doc_type"] for r in corpus} == {"bulletin", "kb", "faq", "runbook"}
    for r in corpus:
        assert {"doc_id", "title", "doc_type", "region", "audience", "allowed_groups"} <= set(r["metadata"])
        assert isinstance(r["metadata"]["allowed_groups"], list)


def test_character_chunks_match_the_guide():
    sizes = [len(c.page_content) for c in chunks()]
    assert (len(sizes), int(np.median(sizes)), max(sizes)) == (55, 472, 597)
    assert all("start_index" in c.metadata for c in chunks())


def test_token_chunks_give_the_same_count():
    """SentenceSplitter counts tokens: 160 tokens ≈ 600 characters, so the counts agree."""
    nodes = SentenceSplitter(chunk_size=160, chunk_overlap=20).get_nodes_from_documents(rk.to_llama_documents(records()))
    assert len(nodes) == 55


def test_format_sources_groups_chunks_by_document():
    meta = {"doc_id": "KB-0610", "title": "AdBlue guide"}
    hits = [Document(page_content="second", metadata={**meta, "start_index": 500}),
            Document(page_content="other", metadata={"doc_id": "TSB-2026-031", "title": "Bulletin", "start_index": 0}),
            Document(page_content="first", metadata={**meta, "start_index": 0})]
    text = format_sources(hits)
    assert text.count('<source id="KB-0610"') == 1 and text.count("<source ") == 2
    assert text.index("first") < text.index("second")


def test_cited_ids_finds_every_id_format():
    answer = "See [TSB-2026-031] and [POL-WAR-001 §3]; also KB-0610, CTR-2026-118 and ENG-SPEC-0417. Not [KB-12]."
    assert rk.cited_ids(answer) == {"TSB-2026-031", "POL-WAR-001", "KB-0610", "CTR-2026-118", "ENG-SPEC-0417"}


class KeywordEmbeddings(Embeddings):
    """A stand-in for the Sentence Transformers model: one dimension per keyword, unit length."""

    WORDS = ("adblue", "dosing", "battery", "charging", "warranty", "ota", "compressor", "vpn")

    def embed_documents(self, texts):
        vectors = np.array([[t.lower().count(w) for w in self.WORDS] + [0.1] for t in texts], dtype="float32")
        return (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def test_faiss_maps_rows_back_to_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(rk, "FAISS_DIR", tmp_path)
    monkeypatch.setattr(part2_faiss, "embeddings", KeywordEmbeddings)
    part2_faiss.faiss_index.cache_clear()
    try:
        hits = retrieve("AdBlue dosing valve")
        assert part2_faiss.faiss_index().ntotal == 55
        assert hits[0].metadata["doc_id"] in {"TSB-2026-031", "KB-0610"} and "score" in hits[0].metadata
        assert (tmp_path / "lab01.faiss").is_file() and (tmp_path / "lab01-chunks.json").is_file()
    finally:
        part2_faiss.faiss_index.cache_clear()
