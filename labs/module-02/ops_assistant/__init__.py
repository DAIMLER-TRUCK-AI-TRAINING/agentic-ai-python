"""Enterprise Operations Assistant: the Module 2 Lab 05 reference solution, packaged for reuse.

    import sys; sys.path.append("../../module-02")   # from labs/module-NN/notebooks
    from ops_assistant import build_assistant

    assistant = build_assistant()                    # provider from LLM_PROVIDER
    result = assistant.handle("Six of us at US-DAL-01 can't connect to the VPN since 08:00.")
    result.record                                    # EscalationRecord, after policy checks
"""

from .assistant import AssistantResult, OpsAssistant, build_assistant, render_system_prompt
from .policy import enforce_policy
from .schemas import ROUTING, EscalationRecord, Triage

__all__ = [
    "AssistantResult",
    "EscalationRecord",
    "OpsAssistant",
    "ROUTING",
    "Triage",
    "build_assistant",
    "enforce_policy",
    "render_system_prompt",
]
