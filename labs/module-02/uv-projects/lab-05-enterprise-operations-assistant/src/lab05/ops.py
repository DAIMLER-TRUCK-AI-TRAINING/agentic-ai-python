"""The packaged tools and schemas in labs/module-02/ops_assistant/, imported in place as the notebook does.

Lab 05 builds on Lab 04's read-only tools and Lab 03's spec; both are packaged there so that Module 5
can import the finished assistant. This project adds the parts you write in the lab.
"""

from __future__ import annotations

import sys

from .common import LAB_DIR

if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

from ops_assistant import tools as ops_tools  # noqa: E402
from ops_assistant.assistant import render_system_prompt  # noqa: E402
from ops_assistant.schemas import MATRIX, ROUTING, EscalationRecord, Triage  # noqa: E402

__all__ = ["MATRIX", "ROUTING", "EscalationRecord", "Triage", "ops_tools", "render_system_prompt"]
