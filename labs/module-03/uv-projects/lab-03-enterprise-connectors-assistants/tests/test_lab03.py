"""Offline tests: the deterministic parts of Lab 03. No API key, model download or network needed.

MockEmbedding stands in for the embedding model: node counts, sync and access trimming do not
depend on what the vectors mean, so the guide's Expected numbers hold with it.
"""

import pytest
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.node_parser import MarkdownNodeParser

from lab03.common import rk
from lab03.part1_connectors import check_part1, load_confluence_export
from lab03.part2_sync import records, sync_runs
from lab03.part3_access import USERS, access_filters
from lab03.part4_assistants import CASES

MOCK = MockEmbedding(embed_dim=8)


@pytest.fixture
def chroma_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(rk, "CHROMA_DIR", tmp_path / "chroma_db")
    return tmp_path


def test_confluence_connector_matches_the_reference():
    check_part1()
    assert load_confluence_export() == rk.load_confluence_export()


def test_retired_versions_are_dropped():
    ids = [r["metadata"]["doc_id"] + "@" + r["metadata"]["version"] for r in records() if r["metadata"]["source"] == "sharepoint"]
    assert len(records()) == 30 and "POL-WAR-001@2025" not in ids and "POL-WAR-001@2026" in ids


def test_access_filters_match_the_reference():
    for user in (*USERS, rk.EU_AFTERSALES):
        for doc_types in (None, ["policy", "contract"]):
            assert access_filters(user, doc_types).model_dump() == rk.access_filters(user, doc_types).model_dump()


def test_sync_reembeds_only_what_changed(chroma_dir):
    store = rk.chroma_vector_store("test-sync")
    runs = sync_runs(rk.to_llama_documents(records()), store, MOCK)
    assert runs == {"first run": 145, "unchanged rerun": 0, "one document edited": 6, "one document deleted": 0}
    assert store._collection.count() == 139


def test_chroma_trims_every_user_to_their_documents(chroma_dir):
    nodes = MarkdownNodeParser().get_nodes_from_documents(rk.to_llama_documents(records()))
    index = VectorStoreIndex(nodes, embed_model=MOCK,
                             storage_context=StorageContext.from_defaults(vector_store=rk.chroma_vector_store("test-acl")))
    for user in (*USERS, rk.EU_AFTERSALES):
        visible = sum(rk.visible_to(user, n.metadata) for n in nodes)
        hits = index.as_retriever(similarity_top_k=len(nodes), filters=access_filters(user)).retrieve("warranty")
        assert len(hits) == visible, user.name
        assert all(rk.visible_to(user, h.node.metadata) for h in hits), user.name
    dealer = set(rk.doc_ids(index.as_retriever(similarity_top_k=len(nodes), filters=access_filters(rk.EU_DEALER)).retrieve("x")))
    assert "POL-WAR-001" in dealer and not dealer & {"POL-GDW-003", "POL-WAR-002", "ENG-SPEC-0440"}


def test_each_assistant_has_an_unanswerable_case():
    for profile, _, questions in CASES.values():
        assert profile.doc_types and sum(not answerable for _, answerable in questions) == 1
