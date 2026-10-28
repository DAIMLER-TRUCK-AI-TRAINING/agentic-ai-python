# `ops_assistant` — the Enterprise Operations Assistant, packaged

The reference solution of [Lab 05](../lab-05-enterprise-operations-assistant.md) as an importable
package. Finish Lab 05 in its notebook first; read this afterwards. **Module 5 Lab 03** imports it
to add guardrails, tracing and an evaluation harness, so its interface is stable:

```python
# (notebook) - from labs/module-NN/notebooks
import sys; sys.path.append("../../module-02")
from ops_assistant import build_assistant

assistant = build_assistant()                      # provider from LLM_PROVIDER; architecture="agent"
result = assistant.handle("Six of us at US-DAL-01 can't connect to the VPN since 08:00.", "INC-104700")
result.record             # EscalationRecord after policy checks (None if the run failed)
result.policy_notes       # every correction policy.enforce_policy made
result.tool_calls, result.input_tokens, result.output_tokens, result.seconds
```

| File | What it holds |
| ---- | ------------- |
| `schemas.py` | `Triage`, `EscalationRecord`, the routing table and the priority matrix from POL-OPS-004 |
| `tools.py` | Read-only tools: `lookup_incident`, `search_kb`, `get_service_status`, `query_ops_db` (validated SELECT on a read-only connection), `web_search` (Tavily or the offline mock), and `langchain_tools()` |
| `policy.py` | `enforce_policy()`: priority and routing consistency, escalation rules E1–E2, unknown KB IDs, reply checks |
| `assistant.py` | `OpsAssistant` with two architectures (`"agent"`, `"workflow"`), and `render_system_prompt()` from the spec |

It needs the course venv and `.env` (see [`labs/setup/`](../../setup/README.md)). The system prompt
is rendered from [`solutions/enterprise-ops-assistant.spec.md`](../solutions/enterprise-ops-assistant.spec.md);
pass `spec_path=` to use your team's spec. The SQLite database is built on first use from
`data/build_ops_db.py`.
