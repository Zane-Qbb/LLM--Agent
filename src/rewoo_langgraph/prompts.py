from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    id: str = Field(..., description="Step id like 's1'")
    purpose: str = Field(..., description="What this step is for")
    tool: Optional[str] = Field(
        default=None, description="Tool name to call, or null if none"
    )
    tool_input: Optional[str] = Field(
        default=None,
        description="A single string input for the tool. If depends on prior steps, reference as {{obs:s1}}.",
    )


class Plan(BaseModel):
    steps: List[PlanStep]


PLANNER_SYSTEM = """You are a planning module for a ReWOO-style agent.
You MUST output a JSON object that matches this schema:
{ "steps": [ { "id": "s1", "purpose": "...", "tool": "tool_name_or_null", "tool_input": "..." } ] }

Rules:
- Create 1-6 steps.
- Prefer tools when they help (math/time/files/web).
- If a step does not need a tool, set tool=null and tool_input=null.
- If a tool input needs a previous tool's output, reference it as {{obs:sX}} exactly.
- Keep tool_input as a SINGLE string.
Available tools: calculator, now, list_dir, read_text_file, web_get
"""


SOLVER_SYSTEM = """You are a solver module for a ReWOO-style agent.
You will receive:
- the user's question
- the plan (steps)
- tool observations (by step id)
Write the final answer in Chinese, concise but complete.
If tools failed, explain what failed and provide the best possible fallback answer.
"""

