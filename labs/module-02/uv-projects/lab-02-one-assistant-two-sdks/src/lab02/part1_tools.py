"""Part 1 — tools as contracts: the tool functions, one neutral definition each, both formats and one dispatcher."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from .common import DATA_DIR, step

# The data behind the tools. The tool functions are read-only and never raise.
INCIDENTS = {i["incident_id"]: i for i in json.loads((DATA_DIR / "incidents.json").read_text(encoding="utf-8"))}
KB = json.loads((DATA_DIR / "kb_articles.json").read_text(encoding="utf-8"))
SERVICES = json.loads((DATA_DIR / "service_status.json").read_text(encoding="utf-8"))
WORD = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def lookup_incident(incident_id: str) -> dict:
    incident_id = incident_id.strip().upper()
    if not re.fullmatch(r"INC-\d{6}", incident_id):
        return {"error": "invalid_id", "incident_id": incident_id, "expected": "INC- followed by 6 digits"}
    return INCIDENTS.get(incident_id) or {"error": "not_found", "incident_id": incident_id}


def search_kb(query: str, max_results: int) -> list[dict]:
    terms = set(WORD.findall(query.lower()))
    scored = []
    for a in KB:
        score = (
            3 * len(terms & set(WORD.findall(" ".join(a["tags"]).lower())))
            + 2 * len(terms & set(WORD.findall(a["title"].lower())))
            + len(terms & set(WORD.findall(a["summary"].lower())))
        )
        if score:
            scored.append((score, a))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["kb_id"]))
    return [
        {"kb_id": a["kb_id"], "title": a["title"], "summary": a["summary"], "escalation": a["escalation"]}
        for _, a in scored[: max(1, min(max_results, 5))]
    ]


def get_service_status(service: str) -> list[dict] | dict:
    needle = service.strip().lower()
    hits = [s for s in SERVICES if needle and (needle in s["ci_id"].lower() or needle in s["service"].lower())]
    return hits or {"error": "not_found", "service": service}


@dataclass(frozen=True)
class ToolSpec:
    """One neutral definition per tool, translated into each provider's format."""

    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]


def object_schema(**properties: dict) -> dict:
    """strict=True needs every property required and no extras."""
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


TOOLS = [
    ToolSpec(
        "lookup_incident",
        "Look up an existing incident by its ID. Returns status, priority, assignment group, CI, related change and updates.",
        object_schema(incident_id={"type": "string", "description": "INC- followed by 6 digits, e.g. INC-104233"}),
        lookup_incident,
    ),
    ToolSpec(
        "search_kb",
        "Search the IT knowledge base by problem description. Returns matching articles with their escalation rule.",
        object_schema(
            query={"type": "string", "description": "Short problem description, e.g. 'vpn rejects new password'"},
            max_results={"type": "integer", "description": "1 to 5; use 3 if unsure"},
        ),
        search_kb,
    ),
    ToolSpec(
        "get_service_status",
        "Current status of a service or configuration item: a CI ID such as 'vpn-gateway-eu-01' or a service name such as 'VPN'.",
        object_schema(service={"type": "string", "description": "CI ID or service name"}),
        get_service_status,
    ),
]
TOOL_BY_NAME = {t.name: t for t in TOOLS}


def to_openai_tool(spec: ToolSpec) -> dict:
    """Responses API function tool: flat, with `parameters`."""
    return {"type": "function", "name": spec.name, "description": spec.description, "parameters": spec.parameters, "strict": True}


def to_claude_tool(spec: ToolSpec) -> dict:
    """Messages API tool: `name`, `description`, the schema under `input_schema`, and `strict`."""
    return {"name": spec.name, "description": spec.description, "input_schema": spec.parameters, "strict": True}


def dispatch(name: str, arguments: str | dict) -> str:
    """Run tool `name` and return its result as a JSON string. Never raises.

    - `arguments` is a JSON string (OpenAI) or an already-parsed dict (Claude).
    - Unknown tool         -> {"error": "unknown_tool", "tool": name}
    - Arguments not JSON   -> {"error": "invalid_json", "detail": ...}
    - Wrong argument names -> {"error": "bad_arguments", "detail": ...}   (a TypeError from the call)
    - Any other exception  -> {"error": "tool_failed", "detail": ...}
    """
    spec = TOOL_BY_NAME.get(name)
    if spec is None:
        return json.dumps({"error": "unknown_tool", "tool": name})
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": "invalid_json", "detail": exc.msg})
    try:
        result = spec.fn(**args)
    except TypeError as exc:
        return json.dumps({"error": "bad_arguments", "detail": str(exc)})
    except Exception as exc:
        return json.dumps({"error": "tool_failed", "detail": f"{exc.__class__.__name__}: {exc}"})
    return json.dumps(result, default=str)


OPENAI_TOOLS = [to_openai_tool(t) for t in TOOLS]
CLAUDE_TOOLS = [to_claude_tool(t) for t in TOOLS]


def run() -> None:
    step("The tool functions, called directly")
    print(json.dumps(lookup_incident("INC-104512"), indent=1)[:400], "…")
    print("search_kb('vpn rejects password', 3) →", [a["kb_id"] for a in search_kb("vpn rejects password", 3)])

    step("One definition, two formats: OpenAI (flat, `parameters`) and Claude (`input_schema`)")
    print(json.dumps(OPENAI_TOOLS[0], indent=1))
    print(json.dumps(CLAUDE_TOOLS[0], indent=1))

    step("The dispatcher both loops share: every failure comes back as data, never as an exception")
    for name, arguments in [
        ("lookup_incident", '{"incident_id": "INC-104512"}'),
        ("reset_password", "{}"),
        ("search_kb", "{not json"),
        ("search_kb", {"q": "vpn"}),
    ]:
        print(f"dispatch({name!r}, {arguments!r}) → {dispatch(name, arguments)[:110]}")
