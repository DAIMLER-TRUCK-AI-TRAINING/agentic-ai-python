"""Shared helpers for the Module 3 labs: corpus loading, connectors, stores, retrieval and metrics.

Each lab writes the new idea itself, in the notebook, then later labs import the reference
version from here, so a lab never depends on whether you finished the previous one's TODOs.

    import sys; sys.path[:0] = ["../../setup", ".."]     # from labs/module-03/notebooks
    import rag_kit as rk

Everything runs locally except the LLM calls: embeddings default to a local Sentence
Transformers model (EMBEDDING_PROVIDER=local), Chroma and FAISS are on disk, and Pinecone is
used only when PINECONE_API_KEY is set.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable

LAB_DIR = Path(__file__).resolve().parent
DATA_DIR = LAB_DIR / "data"
CHROMA_DIR = LAB_DIR / "chroma_db"  # git-ignored (**/chroma_db/)
FAISS_DIR = LAB_DIR / "faiss_index"  # git-ignored (**/faiss_index/)

# Metadata used for filters and citations, but kept out of the text that is embedded.
# Only the title is embedded with each chunk; the LLM sees doc_id and title, so it can cite.
EMBED_KEYS = {"title"}
LLM_KEYS = {"doc_id", "title"}
CITATION = re.compile(r"\b(?:TSB-\d{4}-\d{3}|KB-\d{4}|POL-[A-Z]{3}-\d{3}|CTR-\d{4}-\d{3}|ENG-SPEC-\d{4})\b")


# --------------------------------------------------------------------------- corpus


def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Split ``---`` front matter from a Markdown body. ``allowed_groups`` becomes a list."""
    _, head, body = text.split("---\n", 2)
    meta: dict[str, Any] = {}
    for line in head.strip().splitlines():
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip('"')
    meta["allowed_groups"] = [g.strip() for g in meta.get("allowed_groups", "").split(",") if g.strip()]
    return meta, body.strip()


def load_corpus(root: Path | str = DATA_DIR / "corpus") -> list[dict[str, Any]]:
    """One record per Markdown file: ``{"text": body, "metadata": front matter + source}``."""
    records = []
    for path in sorted(Path(root).rglob("*.md")):
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        records.append({"text": body, "metadata": {**meta, "source": "corpus", "path": str(path.relative_to(LAB_DIR))}})
    return records


def flat_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Chroma (through LlamaIndex) stores only scalar metadata, so an access list becomes one
    flag per group: ``allowed_groups=["dealers_eu"]`` -> ``acl_dealers_eu=1``. (LlamaIndex
    filters take numbers, not booleans.) The same keys work on Pinecone, so one access filter
    serves both stores."""
    flat = {k: v for k, v in meta.items() if k != "allowed_groups"}
    groups = meta.get("allowed_groups", [])
    flat["allowed_groups"] = ", ".join(groups)
    flat.update({f"acl_{group}": 1 for group in groups})
    return flat


def to_llama_documents(records: list[dict[str, Any]]):
    """LlamaIndex Documents with stable ids (the doc_id) and controlled metadata visibility."""
    from llama_index.core import Document

    docs = []
    for r in records:
        meta = flat_metadata(r["metadata"])
        docs.append(
            Document(
                text=r["text"],
                metadata=meta,
                id_=meta["doc_id"],
                excluded_embed_metadata_keys=[k for k in meta if k not in EMBED_KEYS],
                excluded_llm_metadata_keys=[k for k in meta if k not in LLM_KEYS],
            )
        )
    return docs


# --------------------------------------------------------------------------- connectors


class _StorageToText(HTMLParser):
    """Confluence storage format (XHTML) to Markdown-like text: headings, paragraphs, tables."""

    def __init__(self) -> None:
        super().__init__()
        self.out: list[str] = []
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"h1", "h2", "h3"}:
            self.out.append("\n" + "#" * (int(tag[1]) + 1) + " ")
        elif tag == "tr":
            self.row = []
        elif tag in {"td", "th"}:
            self.cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.row is not None and self.cell is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.out.append("| " + " | ".join(self.row) + " |\n")
            self.row = None
        elif tag in {"p", "h1", "h2", "h3", "table"}:
            self.out.append("\n")

    def handle_data(self, data: str) -> None:
        (self.cell if self.cell is not None else self.out).append(data)


def storage_to_text(html: str) -> str:
    parser = _StorageToText()
    parser.feed(html)
    return re.sub(r"\n{3,}", "\n\n", "".join(parser.out)).strip()


def load_sharepoint_export(root: Path | str = DATA_DIR / "connectors" / "sharepoint") -> list[dict[str, Any]]:
    """Records from the SharePoint export: file text plus the library's columns and permissions."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    records = []
    for item in manifest["items"]:
        text = (root / manifest["folder"] / item["file"]).read_text(encoding="utf-8")
        col = item["columns"]
        records.append({"text": text, "metadata": {
            "doc_id": col["DocId"], "title": text.splitlines()[0].lstrip("# ").strip(),
            "doc_type": col["DocType"], "region": col["Region"], "audience": col["Audience"],
            "owner_team": col["Owner"], "language": "en", "updated": item["lastModified"][:10],
            "status": col["Status"], "version": col["Version"], "allowed_groups": item["permissions"],
            "source": "sharepoint", "url": item["webUrl"], "product": "ALL",
        }})
    return records


def load_confluence_export(path: Path | str = DATA_DIR / "connectors" / "confluence" / "pages.json") -> list[dict[str, Any]]:
    """Records from the Confluence export (reference solution of Lab 03 Part 1)."""
    export = json.loads(Path(path).read_text(encoding="utf-8"))
    records = []
    for page in export["results"]:
        doc_id, _, title = page["title"].partition(" ")
        records.append({"text": f"# {page['title']}\n\n" + storage_to_text(page["body"]["storage"]["value"]), "metadata": {
            "doc_id": doc_id, "title": title, "doc_type": "engineering", "region": "ALL",
            "audience": "internal", "owner_team": "Engineering", "language": "en",
            "updated": page["version"]["when"][:10], "status": "published", "version": str(page["version"]["number"]),
            "allowed_groups": page["restrictions"]["read"]["group"], "source": "confluence",
            "url": f"https://confluence.example/wiki{page['_links']['webui']}", "product": "ALL",
        }})
    return records


def load_all_sources(include_retired: bool = False) -> list[dict[str, Any]]:
    """The unified knowledge base of Labs 03-04: corpus + SharePoint + Confluence."""
    records = load_corpus() + load_sharepoint_export() + load_confluence_export()
    return [r for r in records if include_retired or r["metadata"].get("status", "published") == "published"]


# --------------------------------------------------------------------------- models and stores


def setup_llama(provider: str | None = None, *, small: bool = False):
    """Set LlamaIndex's global LLM and embedding model from .env; return the LLM."""
    from llama_index.core import Settings
    from llm import get_llama_embed_model

    Settings.embed_model = get_llama_embed_model()
    Settings.llm = llama_llm(provider, small=small)
    return Settings.llm


def llama_llm(provider: str | None = None, *, small: bool = False):
    """LlamaIndex LLM for a provider, with low reasoning effort on OpenAI's own API."""
    from llm import get_llama_llm, get_provider

    name = get_provider(provider)
    kwargs = {"reasoning_options": {"effort": "low"}} if name == "openai" else {}
    return get_llama_llm(name, small=small, **kwargs)


def chroma_vector_store(name: str, *, fresh: bool = True):
    """A LlamaIndex ChromaVectorStore on disk; ``fresh`` drops the collection first, so a rerun
    does not add every chunk twice."""
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if fresh and name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)
    collection = client.get_or_create_collection(name, configuration={"hnsw": {"space": "cosine"}})
    return ChromaVectorStore(chroma_collection=collection)


def pinecone_vector_store(index_name: str, dimension: int, namespace: str):
    """A LlamaIndex PineconeVectorStore on a serverless index (created if missing), namespace emptied."""
    from llama_index.vector_stores.pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name, dimension=dimension, metric="cosine",
            spec=ServerlessSpec(cloud=os.getenv("PINECONE_CLOUD", "aws"), region=os.getenv("PINECONE_REGION", "us-east-1")),
        )
    index = pc.Index(index_name)
    if index.describe_index_stats().get("dimension") != dimension:
        raise ValueError(f"Pinecone index {index_name!r} has another dimension; set PINECONE_INDEX to a new name")
    try:
        index.delete(delete_all=True, namespace=namespace)
    except Exception:  # the namespace does not exist yet
        pass
    return PineconeVectorStore(pinecone_index=index, namespace=namespace)


def wait_for_pinecone(vector_store, expected: int, timeout: float = 60.0) -> float:
    """Pinecone serverless is eventually consistent: wait until the namespace holds every vector."""
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        stats = vector_store._pinecone_index.describe_index_stats()
        ns = (stats.get("namespaces") or {}).get(vector_store.namespace or "")
        if ns and ns.get("vector_count", 0) >= expected:
            return time.perf_counter() - started
        time.sleep(2)
    raise TimeoutError(f"Pinecone namespace still has fewer than {expected} vectors after {timeout:.0f}s")


def vector_backend() -> str:
    """``pinecone`` when PINECONE_API_KEY is set, otherwise ``chroma`` (the offline fallback)."""
    return "pinecone" if os.getenv("PINECONE_API_KEY", "").strip() else "chroma"


def build_index(nodes: list, name: str, *, backend: str = "chroma", namespace: str = "default"):
    """Embed ``nodes`` into a fresh vector store and return a VectorStoreIndex over it."""
    from llama_index.core import Settings, StorageContext, VectorStoreIndex

    if backend == "pinecone":
        dimension = len(Settings.embed_model.get_text_embedding("dimension probe"))
        store = pinecone_vector_store(os.getenv("PINECONE_INDEX", "").strip() or name[:45], dimension, namespace)
    else:
        store = chroma_vector_store(name)
    index = VectorStoreIndex(nodes, storage_context=StorageContext.from_defaults(vector_store=store))
    if backend == "pinecone":
        wait_for_pinecone(store, len(nodes))
    return index


# --------------------------------------------------------------------------- access control


@dataclass(frozen=True)
class User:
    """The authenticated caller. Access filters come from here, never from the model."""

    name: str
    region: str
    groups: tuple[str, ...]
    audiences: tuple[str, ...] = ("internal", "dealer", "customer")


EU_DEALER = User("eu_dealer_advisor", "EU", ("dealers_eu",), ("dealer", "customer"))
US_WARRANTY = User("us_warranty_analyst", "US", ("warranty_us", "aftersales_us", "all_employees"))
FLEET_CUSTOMER = User("fleet_customer", "EU", (), ("customer",))
EU_AFTERSALES = User("eu_aftersales_engineer", "EU", ("aftersales_eu", "warranty_eu", "engineering", "all_employees"))


def access_filters(user: User, doc_types: list[str] | None = None):
    """MetadataFilters: region in (user region, ALL) AND audience allowed AND any of the user's groups."""
    from llama_index.core.vector_stores import FilterCondition, FilterOperator, MetadataFilter, MetadataFilters

    groups = [MetadataFilter(key=f"acl_{g}", value=1) for g in (*user.groups, "public")]
    filters = [
        MetadataFilter(key="region", value=[user.region, "ALL"], operator=FilterOperator.IN),
        MetadataFilter(key="audience", value=list(user.audiences), operator=FilterOperator.IN),
        MetadataFilters(filters=groups, condition=FilterCondition.OR),
    ]
    if doc_types:
        filters.append(MetadataFilter(key="doc_type", value=list(doc_types), operator=FilterOperator.IN))
    return MetadataFilters(filters=filters, condition=FilterCondition.AND)


def visible_to(user: User, meta: dict[str, Any]) -> bool:
    """The same rule as ``access_filters``, in Python, for retrievers that cannot filter (BM25)."""
    return (
        meta.get("region") in {user.region, "ALL"}
        and meta.get("audience") in user.audiences
        and any(meta.get(f"acl_{g}") for g in (*user.groups, "public"))
    )


# --------------------------------------------------------------------------- retrieval


def doc_ids(nodes) -> list[str]:
    """The doc_id of each retrieved node, in rank order (duplicates kept: one per chunk)."""
    return [n.node.metadata["doc_id"] if hasattr(n, "node") else n.metadata["doc_id"] for n in nodes]


def bm25_retriever(nodes: list, k: int = 10):
    """BM25 keyword retrieval over the nodes (English stemming; codes like FC-ADB-0412 match exactly)."""
    import Stemmer
    from llama_index.retrievers.bm25 import BM25Retriever

    return BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=k, stemmer=Stemmer.Stemmer("english"), language="english")


def hybrid_retriever(index, nodes: list, k: int = 4, pool: int = 10, filters=None):
    """Vector + BM25 fused with reciprocal rank fusion; ``num_queries=1`` means no LLM query rewriting."""
    from llama_index.core import Settings
    from llama_index.core.retrievers import QueryFusionRetriever

    return QueryFusionRetriever(
        [index.as_retriever(similarity_top_k=pool, filters=filters), bm25_retriever(nodes, pool)],
        similarity_top_k=k, num_queries=1, mode="reciprocal_rerank", use_async=False, llm=Settings.llm,
    )


def reranker(top_n: int = 4):
    """Local cross-encoder re-ranker (RERANK_MODEL, English-only by default). It runs on the CPU:
    a small model is fast there, and the Apple GPU backend can crash in an agent's worker thread."""
    from llama_index.core.postprocessor import SentenceTransformerRerank

    model = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    return SentenceTransformerRerank(model=model, top_n=top_n, device="cpu")


# --------------------------------------------------------------------------- evaluation


def retrieval_metrics(ids: list[str], relevant: set[str], k: int) -> dict[str, float]:
    """Precision@k, recall@k, hit rate@k and reciprocal rank for one query (reference for Lab 02)."""
    ids = ids[:k]
    first = next((rank for rank, i in enumerate(ids, 1) if i in relevant), None)
    return {
        "precision": sum(i in relevant for i in ids) / k,
        "recall": len(relevant & set(ids)) / len(relevant),
        "hit_rate": float(first is not None),
        "mrr": 1 / first if first else 0.0,
    }


def evaluate(retrieve_ids: Callable[[str], list[str]], queries: list[dict], k: int = 4) -> dict[str, float]:
    """Mean metrics over the answerable labelled queries."""
    rows = [retrieval_metrics(retrieve_ids(q["query"]), set(q["relevant"]), k) for q in queries if q["relevant"]]
    return {m: round(sum(r[m] for r in rows) / len(rows), 2) for m in rows[0]}


def cited_ids(text: str) -> set[str]:
    """Every document ID cited in an answer, for example [KB-0610] or [POL-WAR-001 §3]."""
    return set(CITATION.findall(text))


@dataclass
class Timer:
    """``with Timer() as t: ...; t.seconds``"""

    seconds: float = field(default=0.0)

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = round(time.perf_counter() - self._start, 2)


def agent_tool(fn: Callable, name: str):
    """A FunctionTool whose async path calls ``fn`` directly on the event loop, instead of in a worker
    thread: local PyTorch models (embeddings, re-ranker) are not safe to run from a thread pool on every
    platform."""
    from llama_index.core.tools import FunctionTool

    async def run_inline(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    return FunctionTool.from_defaults(fn=fn, async_fn=run_inline, name=name)


def todo_or_reference(fn: Callable, reference: Callable, *probe: Any) -> Callable:
    """Use your TODO function once it works; until then, the reference from rag_kit (with a note).
    This keeps a starter notebook runnable from top to bottom."""
    try:
        fn(*probe)
    except NotImplementedError:
        print(f"⏳ {fn.__name__}: TODO not done yet, so this cell uses rag_kit.{reference.__name__} for now")
        return reference
    return fn


# --------------------------------------------------------------------------- assistants and agents (Labs 03-04)


def build_knowledge_base(name: str, *, backend: str = "chroma"):
    """Chunk the unified sources by Markdown section and index them; return (index, nodes)."""
    from llama_index.core.node_parser import MarkdownNodeParser

    nodes = MarkdownNodeParser().get_nodes_from_documents(to_llama_documents(load_all_sources()))
    return build_index(nodes, name, backend=backend), nodes


@dataclass(frozen=True)
class RagProfile:
    """What one assistant searches, how many chunks it keeps, its score floor and its answer style."""

    name: str
    doc_types: tuple[str, ...]
    instructions: str
    k: int = 4
    floor: float = -1.0  # minimum cross-encoder score; calibrated on labelled questions (Lab 03 Part 5)


ANSWER_RULES = (
    "Answer only from the sources below. Cite the source id in square brackets after each claim, "
    "for example [POL-WAR-001 §3] or [KB-0701]. If the sources do not answer the question, say "
    "that you cannot find it in the available documents."
)


def make_assistant(profile: RagProfile, user: User, index, llm=None):
    """One retrieval-and-answer pattern (reference for Lab 03 Part 4): access and doc-type filters
    first, then vector search, a cross-encoder re-rank, the score floor, and the profile's prompt.
    When nothing passes the floor, LlamaIndex answers "Empty Response" without calling the model."""
    from llama_index.core import PromptTemplate
    from llama_index.core.postprocessor import SimilarityPostprocessor

    prompt = PromptTemplate(
        f"{profile.instructions}\n{ANSWER_RULES}\n\n<sources>\n{{context_str}}\n</sources>\n\nQuestion: {{query_str}}\nAnswer: "
    )
    return index.as_query_engine(
        llm=llm,
        similarity_top_k=2 * profile.k,
        filters=access_filters(user, list(profile.doc_types)),
        node_postprocessors=[reranker(profile.k), SimilarityPostprocessor(similarity_cutoff=profile.floor)],
        text_qa_template=prompt,
    )


def answered(response) -> bool:
    """False when the query engine found nothing above the floor (no model call was made)."""
    return bool(response.source_nodes) and str(response).strip() != "Empty Response"


def make_search_tool(index, nodes: list, user: User, *, k: int = 4, floor: float = -1.0) -> Callable:
    """``search_knowledge`` for an agent (reference for Lab 04 Part 1): hybrid search, re-rank,
    floor, the caller's access filters, and JSON passages with stable doc_ids."""
    from llama_index.core import QueryBundle

    allowed = [n for n in nodes if visible_to(user, n.metadata)]
    rerank = reranker(k)

    def search_knowledge(query: str, doc_type: str = "any") -> str:
        """Search Daimler Truck workshop bulletins (bulletin), repair guides (kb), customer FAQs (faq),
        IT runbooks (runbook), policies (policy), contracts (contract) and engineering specs
        (engineering). Use it for any question about vehicles, fault codes, repairs, warranty,
        policies or IT procedures; do not use it for greetings or thanks. doc_type narrows the
        search to one of those types, or "any". Returns up to 4 passages as JSON with doc_id and title."""
        types = None if doc_type == "any" else [doc_type]
        pool = [n for n in allowed if types is None or n.metadata["doc_type"] in types]
        if not pool:
            return f"NO_RESULTS: unknown doc_type {doc_type!r}; use one of the listed types or 'any'."
        candidates = hybrid_retriever(index, pool, k=10, pool=10, filters=access_filters(user, types)).retrieve(query)
        hits = [h for h in rerank.postprocess_nodes(candidates, query_bundle=QueryBundle(query)) if h.score >= floor]
        if not hits:
            return "NO_RESULTS: nothing relevant found. Rephrase the query, or try doc_type='any'."
        return json.dumps([{"doc_id": h.node.metadata["doc_id"], "title": h.node.metadata["title"],
                            "text": h.node.get_content()} for h in hits], ensure_ascii=False)

    return search_knowledge
