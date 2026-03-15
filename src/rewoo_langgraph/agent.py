from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from .prompts import PLANNER_SYSTEM, SOLVER_SYSTEM, Plan
from .tools import build_tools


class ReWOOState(TypedDict, total=False):
    question: str
    plan: Dict[str, Any]
    observations: Dict[str, str]
    answer: str


def _get_llm() -> ChatOpenAI:
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set env vars OPENAI_API_KEY (and optionally OPENAI_BASE_URL/OPENAI_MODEL)."
        )
    kwargs: Dict[str, Any] = {"model": model, "temperature": 0}
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


_OBS_REF_RE = re.compile(r"\{\{obs:(s\d+)\}\}")


def _render_tool_input(template: Optional[str], observations: Dict[str, str]) -> Optional[str]:
    if template is None:
        return None

    def repl(m: re.Match) -> str:
        sid = m.group(1)
        return observations.get(sid, f"<MISSING_OBS:{sid}>")

    return _OBS_REF_RE.sub(repl, template)


def build_graph(workspace_root: Optional[str] = None):
    workspace_root = workspace_root or os.getcwd()
    tools = build_tools(workspace_root)
    tool_map = {t.name: t for t in tools}
    llm = _get_llm()

    async def plan_node(state: ReWOOState) -> ReWOOState:
        q = state["question"]
        msg = [
            SystemMessage(content=PLANNER_SYSTEM),
            HumanMessage(content=q),
        ]
        out = await llm.ainvoke(msg)
        text = out.content if isinstance(out.content, str) else json.dumps(out.content)
        # Best-effort JSON parse: extract first {...}
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise ValueError(f"Planner did not return JSON. Raw: {text[:500]}")
        raw = json.loads(m.group(0))
        plan = Plan.model_validate(raw).model_dump()
        # log for learning ReWOO
        print("\n[planner] generated plan:")
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return {"plan": plan, "observations": {}}

    async def act_node(state: ReWOOState) -> ReWOOState:
        plan = state.get("plan") or {"steps": []}
        observations = dict(state.get("observations") or {})
        for step in plan.get("steps", []):
            sid = step["id"]
            tool_name = step.get("tool")
            tool_input = step.get("tool_input")
            if not tool_name:
                observations[sid] = ""
                continue
            tool = tool_map.get(tool_name)
            if not tool:
                observations[sid] = f"UNKNOWN_TOOL: {tool_name}"
                continue
            rendered = _render_tool_input(tool_input, observations) or ""
            print(f"\n[act] step={sid} tool={tool_name} input={rendered!r}")
            try:
                result = await tool.ainvoke(rendered)
            except Exception as e:
                observations[sid] = f"TOOL_ERROR({tool_name}): {e}"
                print(f"[act] step={sid} tool={tool_name} ERROR: {e}")
            else:
                observations[sid] = str(result)
                print(f"[act] step={sid} tool={tool_name} output (truncated 500 chars):")
                print(str(result)[:500])
        return {"observations": observations}

    async def solve_node(state: ReWOOState) -> ReWOOState:
        q = state["question"]
        plan = state.get("plan") or {}
        observations = state.get("observations") or {}
        payload = {
            "question": q,
            "plan": plan,
            "observations": observations,
        }
        msg = [
            SystemMessage(content=SOLVER_SYSTEM),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2)),
        ]
        out = await llm.ainvoke(msg)
        answer = out.content if isinstance(out.content, str) else json.dumps(out.content, ensure_ascii=False)
        return {"answer": answer}

    g = StateGraph(ReWOOState)
    g.add_node("plan", plan_node)
    g.add_node("act", act_node)
    g.add_node("solve", solve_node)
    g.set_entry_point("plan")
    g.add_edge("plan", "act")
    g.add_edge("act", "solve")
    g.add_edge("solve", END)
    return g.compile()

