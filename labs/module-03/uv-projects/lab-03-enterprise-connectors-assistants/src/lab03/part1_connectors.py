"""Part 1 — SharePoint and Confluence exports into records with the shared metadata schema."""

from __future__ import annotations

import functools
import importlib.util
import json
import os
from pathlib import Path

import pandas as pd

from .common import DATA_DIR, check, rk, show, step

# Live readers when the tenant is configured (instructor), otherwise the exports. No credentials in code.
LIVE = {
    "SharePoint": ("llama_index.readers.microsoft_sharepoint", ["SHAREPOINT_CLIENT_ID", "SHAREPOINT_CLIENT_SECRET", "SHAREPOINT_TENANT_ID"]),
    "Confluence": ("llama_index.readers.confluence", ["CONFLUENCE_URL", "CONFLUENCE_API_TOKEN"]),
}


def load_confluence_export(path=DATA_DIR / "connectors" / "confluence" / "pages.json") -> list[dict]:
    """One record per page: {"text": ..., "metadata": {...}}.

    - text: "# <page title>\\n\\n" + the storage-format body converted with rk.storage_to_text(...)
    - metadata: doc_id and title (split the page title at the first space: "ENG-SPEC-0417 ED-26 ..."),
      doc_type="engineering", region="ALL", audience="internal", owner_team="Engineering", language="en",
      updated (the date part of version.when), status="published", version (str of version.number),
      allowed_groups (restrictions.read.group, a list), source="confluence",
      url ("https://confluence.example/wiki" + _links.webui), product="ALL"
    """
    export = json.loads(Path(path).read_text(encoding="utf-8"))
    records = []
    for page in export["results"]:
        doc_id, _, title = page["title"].partition(" ")
        records.append({"text": f"# {page['title']}\n\n" + rk.storage_to_text(page["body"]["storage"]["value"]), "metadata": {
            "doc_id": doc_id, "title": title, "doc_type": "engineering", "region": "ALL", "audience": "internal",
            "owner_team": "Engineering", "language": "en", "updated": page["version"]["when"][:10],
            "status": "published", "version": str(page["version"]["number"]),
            "allowed_groups": page["restrictions"]["read"]["group"], "source": "confluence",
            "url": f"https://confluence.example/wiki{page['_links']['webui']}", "product": "ALL",
        }})
    return records


@functools.cache
def sharepoint() -> list[dict]:
    return rk.load_sharepoint_export()


@functools.cache
def confluence() -> list[dict]:
    return load_confluence_export()


def check_part1():
    pages = load_confluence_export()
    assert [p["metadata"]["doc_id"] for p in pages] == ["ENG-SPEC-0417", "ENG-SPEC-0422", "ENG-SPEC-0440"]
    assert set(pages[0]["metadata"]) >= set(sharepoint()[0]["metadata"]) - {"url"}
    assert pages[2]["metadata"]["allowed_groups"] == ["engineering"]
    assert "<td>" not in pages[0]["text"] and "| Cell temperature spread | > 8 K |" in pages[0]["text"]


def run() -> None:
    step("Live readers or offline exports")
    for source, (module, variables) in LIVE.items():
        live = importlib.util.find_spec(module.rsplit(".", 1)[0]) is not None and all(os.getenv(v) for v in variables)
        print(f"{source:10}: {'live reader available (instructor demo)' if live else 'offline export in data/connectors/ (no tenant configured)'}")

    step("SharePoint: the library's columns and permissions become metadata")
    columns = ("doc_id", "version", "status", "region", "audience", "allowed_groups")
    show(pd.DataFrame([{k: r["metadata"][k] for k in columns} for r in sharepoint()]), index=False)

    step("Confluence: storage-format XHTML to text, page restrictions to allowed_groups")
    check("Part 1 — Confluence connector", check_part1)
    print(confluence()[0]["text"][:400])
