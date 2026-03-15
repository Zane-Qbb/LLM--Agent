from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.base import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from rewoo_langgraph.tools import build_tools

from .prompts import EXECUTOR_SYSTEM, JUDGE_SYSTEM, PLANNER_SYSTEM, REPLANNER_SYSTEM, JudgeDecision, Plan, ReplanDecision


class PlanExecuteState(TypedDict, total=False):
    objective: str
    original_plan: List[str]
    plan: List[str]  # remaining steps
    past_steps: List[str]  # executed steps (with results)
    answer: str
    _last_task: str
    _last_result: str
    _decision: str  # continue|replan|final


def _get_llm(env_model_key: str = "OPENAI_MODEL", default: str = "gpt-4.1-mini") -> ChatOpenAI:
    model = os.getenv(env_model_key, default)
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set env vars OPENAI_API_KEY (and optionally OPENAI_BASE_URL/OPENAI_MODEL)."
        )
    timeout_s = float(os.getenv("OPENAI_TIMEOUT_S", "120"))
    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
    kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "timeout": timeout_s,
        "max_retries": max_retries,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def _extract_json(text: str) -> Dict[str, Any]:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"Model did not return JSON. Raw: {text[:500]}")
    return json.loads(m.group(0))


def _stream_enabled() -> bool:
    return os.getenv("STREAM_LLM", "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _ts() -> str:
    # lightweight timestamp without extra deps
    try:
        import datetime as dt

        return dt.datetime.now().strftime("%H:%M:%S")
    except Exception:
        return ""


_STEP_PREFIX_RE = re.compile(r"^\s*\d+\.\s*")


def _normalize_steps(steps: List[str]) -> List[str]:
    out: List[str] = []
    for s in steps:
        s2 = _STEP_PREFIX_RE.sub("", (s or "").strip())
        if s2:
            out.append(s2)
    return out


async def _ainvoke_text(llm: ChatOpenAI, messages: List[BaseMessage], *, label: str) -> str:
    """
    Invoke an LLM and return text content. If STREAM_LLM=1, stream tokens to stdout.
    """
    if not _stream_enabled():
        print(f"\n[{label}] start {(_ts() or '').strip()} (stream=0)")
        out = await llm.ainvoke(messages)
        return out.content if isinstance(out.content, str) else json.dumps(out.content, ensure_ascii=False)

    print(f"\n[{label}] start {(_ts() or '').strip()} (stream=1)")
    parts: List[str] = []
    async for chunk in llm.astream(messages):
        delta = chunk.content if isinstance(chunk.content, str) else ""
        if delta:
            print(delta, end="", flush=True)
            parts.append(delta)
    print("", flush=True)
    return "".join(parts)


async def _run_react_executor(
    *,
    llm: ChatOpenAI,
    tools: List[Any],
    tool_map: Dict[str, Any],
    task: str,
    context: str,
    max_iters: int = 8,
) -> str:
    """
    Minimal ReAct-style loop using tool calling.
    We rely on OpenAI-compatible function/tool calling via `bind_tools`.
    """
    bound = llm.bind_tools(tools)
    messages = [
        SystemMessage(content=EXECUTOR_SYSTEM),
        HumanMessage(
            content=(
                "CURRENT TASK:\n"
                f"{task}\n\n"
                "CONTEXT (may include prior results):\n"
                f"{context}\n\n"
                "Please execute the task and return a concise completion note."
            )
        ),
    ]
    print(f"\n[executor] prompt: {messages}")
    had_tool_error = False
    for _ in range(max_iters):
        out = await bound.ainvoke(messages)
        messages.append(out)

        tool_calls = getattr(out, "tool_calls", None) or []
        if not tool_calls:
            content = out.content if isinstance(out.content, str) else json.dumps(out.content, ensure_ascii=False)
            return content.strip()

        for call in tool_calls:
            name = call.get("name")
            args = call.get("args")
            tool = tool_map.get(name)
            if not tool:
                messages.append(ToolMessage(content=f"UNKNOWN_TOOL: {name}", tool_call_id=call.get("id")))
                continue
            try:
                result = await tool.ainvoke(args)
            except Exception as e:
                had_tool_error = True
                messages.append(
                    ToolMessage(content=f"TOOL_ERROR({name}): {e}", tool_call_id=call.get("id"))
                )
            else:
                messages.append(ToolMessage(content=str(result), tool_call_id=call.get("id")))

    # If executor doesn't finish, return best-effort summary.
    tail = "Executor reached max tool iterations without a final response. Provide partial progress based on context."
    if had_tool_error:
        tail = "TOOL_ERROR: " + tail
    return tail


def build_graph(workspace_root: Optional[str] = None):
    workspace_root = workspace_root or os.getcwd()
    tools = build_tools(workspace_root)
    tool_map = {t.name: t for t in tools}

    planner_llm = _get_llm("PLANNER_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    # allow smaller executor model if desired
    executor_llm = _get_llm("EXECUTOR_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    judge_llm = _get_llm("JUDGE_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    replanner_llm = _get_llm("REPLANNER_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))

    async def planner_node(state: PlanExecuteState) -> PlanExecuteState:
        objective = state["objective"]
        msg = [SystemMessage(content=PLANNER_SYSTEM), HumanMessage(content=objective)]
        print(f"\n[planner] prompt: {msg}")
        text = await _ainvoke_text(planner_llm, msg, label="planner")
        raw = _extract_json(text)
        plan_obj = Plan.model_validate(raw)
        steps = _normalize_steps(list(plan_obj.steps))
        print("\n[planner] plan:")
        print(json.dumps({"steps": steps}, ensure_ascii=False, indent=2))
        return {"original_plan": steps, "plan": steps, "past_steps": []}

    async def executor_node(state: PlanExecuteState) -> PlanExecuteState:
        plan = list(state.get("plan") or [])
        if not plan:
            return {}

        task = plan[0]
        past_steps = list(state.get("past_steps") or [])
        # context is only the executed steps (kept minimal to reduce cost)
        context = "\n".join(past_steps[-8:]) if past_steps else "(none)"
        result = await _run_react_executor(
            llm=executor_llm,
            tools=tools,
            tool_map=tool_map,
            task=task,
            context=context,
        )
        entry = f"TASK: {task}\nRESULT: {result}".strip()
        print(f"\n[executor] executed step: {entry}")
        past_steps.append(entry)
        # pop executed step; remaining stays as-is unless replanner updates it
        remaining = plan[1:]
        return {
            "plan": remaining,
            "past_steps": past_steps,
            "_last_task": task,
            "_last_result": result,
        }

    async def judge_node(state: PlanExecuteState) -> PlanExecuteState:
        objective = state["objective"]
        last_task = state.get("_last_task") or ""
        last_result = state.get("_last_result") or ""
        remaining = state.get("plan") or []

        payload = {
            "objective": objective,
            "last_task": last_task,
            "last_result": last_result,
            "remaining_steps": remaining,
        }
        msg = [
            SystemMessage(content=JUDGE_SYSTEM),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2)),
        ]
        print(f"\n[judge] prompt: {json.dumps(payload, ensure_ascii=False, indent=2)}")
        text = await _ainvoke_text(judge_llm, msg, label="judge")
        raw = _extract_json(text)
        decision = JudgeDecision.model_validate(raw)

        if decision.action == "final":
            ans = (decision.answer or "").strip()
            return {"answer": ans, "_decision": "final"}
        return {"_decision": decision.action}

    async def replanner_node(state: PlanExecuteState) -> PlanExecuteState:
        objective = state["objective"]
        original_plan = state.get("original_plan") or []
        past_steps = state.get("past_steps") or []
        remaining = state.get("plan") or []

        prompt = REPLANNER_SYSTEM.format(
            objective=objective,
            plan=json.dumps(original_plan, ensure_ascii=False, indent=2),
            past_steps=json.dumps(past_steps, ensure_ascii=False, indent=2),
            remaining_steps=json.dumps(remaining, ensure_ascii=False, indent=2),
        )
        print(f"\n[replanner] prompt: {prompt}")
        msg = [SystemMessage(content=prompt), HumanMessage(content="Update plan now.")]
        text = await _ainvoke_text(replanner_llm, msg, label="replanner")
        raw = _extract_json(text)
        decision = ReplanDecision.model_validate(raw)

        if decision.action == "final":
            answer = (decision.answer or "").strip()
            return {"answer": answer, "plan": []}

        # action == "plan"
        steps = _normalize_steps(list(decision.steps or []))
        # If replanner returns empty steps but we still have remaining, keep remaining as fallback.
        if not steps and remaining:
            steps = list(remaining)
        return {"plan": steps}

    def router_after_plan(state: PlanExecuteState) -> str:
        plan = state.get("plan") or []
        return "execute" if plan else "end"

    def router_after_judge(state: PlanExecuteState) -> str:
        if state.get("answer"):
            return "end"
        d = (state.get("_decision") or "").strip().lower()
        if d == "replan":
            return "replan"
        if d == "continue":
            # if no remaining steps, we should end (replanner can also finalize earlier, but this is a safe default)
            return "execute" if (state.get("plan") or []) else "end"
        if d == "final":
            return "end"
        # fallback: be conservative and replan
        return "replan"

    g = StateGraph(PlanExecuteState)
    g.add_node("plan", planner_node)
    g.add_node("execute", executor_node)
    g.add_node("judge", judge_node)
    g.add_node("replan", replanner_node)

    g.set_entry_point("plan")
    # Plan & Execute with a judge (no replanning unless needed):
    # plan -> execute -> judge -> (execute next | replan | end)
    g.add_conditional_edges("plan", router_after_plan, {"execute": "execute", "end": END})
    g.add_edge("execute", "judge")
    g.add_conditional_edges("judge", router_after_judge, {"execute": "execute", "replan": "replan", "end": END})
    g.add_edge("replan", "execute")
    return g.compile()


def build_executor_graph(workspace_root: Optional[str] = None):
    """
    Graph that assumes the plan is already confirmed/edited by a human.
    Entry point is `execute` instead of `plan`.
    """
    workspace_root = workspace_root or os.getcwd()
    tools = build_tools(workspace_root)
    tool_map = {t.name: t for t in tools}

    # Reuse same models as main graph
    executor_llm = _get_llm("EXECUTOR_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    judge_llm = _get_llm("JUDGE_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    replanner_llm = _get_llm("REPLANNER_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))

    async def executor_node(state: PlanExecuteState) -> PlanExecuteState:
        plan = list(state.get("plan") or [])
        if not plan:
            return {}

        task = plan[0]
        past_steps = list(state.get("past_steps") or [])
        context = "\n".join(past_steps[-8:]) if past_steps else "(none)"
        result = await _run_react_executor(
            llm=executor_llm,
            tools=tools,
            tool_map=tool_map,
            task=task,
            context=context,
        )
        entry = f"TASK: {task}\nRESULT: {result}".strip()
        print(f"\n[executor] executed step: {entry}")
        past_steps.append(entry)
        remaining = plan[1:]
        return {
            "plan": remaining,
            "past_steps": past_steps,
            "_last_task": task,
            "_last_result": result,
        }

    async def judge_node(state: PlanExecuteState) -> PlanExecuteState:
        objective = state["objective"]
        last_task = state.get("_last_task") or ""
        last_result = state.get("_last_result") or ""
        remaining = state.get("plan") or []

        payload = {
            "objective": objective,
            "last_task": last_task,
            "last_result": last_result,
            "remaining_steps": remaining,
        }
        msg = [
            SystemMessage(content=JUDGE_SYSTEM),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2)),
        ]
        text = await _ainvoke_text(judge_llm, msg, label="judge")
        raw = _extract_json(text)
        decision = JudgeDecision.model_validate(raw)

        if decision.action == "final":
            ans = (decision.answer or "").strip()
            return {"answer": ans, "_decision": "final"}
        return {"_decision": decision.action}

    async def replanner_node(state: PlanExecuteState) -> PlanExecuteState:
        objective = state["objective"]
        original_plan = state.get("original_plan") or []
        past_steps = state.get("past_steps") or []
        remaining = state.get("plan") or []

        prompt = REPLANNER_SYSTEM.format(
            objective=objective,
            plan=json.dumps(original_plan, ensure_ascii=False, indent=2),
            past_steps=json.dumps(past_steps, ensure_ascii=False, indent=2),
            remaining_steps=json.dumps(remaining, ensure_ascii=False, indent=2),
        )
        msg = [SystemMessage(content=prompt), HumanMessage(content="Update plan now.")]
        text = await _ainvoke_text(replanner_llm, msg, label="replanner")
        raw = _extract_json(text)
        decision = ReplanDecision.model_validate(raw)

        if decision.action == "final":
            answer = (decision.answer or "").strip()
            return {"answer": answer, "plan": []}

        steps = _normalize_steps(list(decision.steps or []))
        if not steps and remaining:
            steps = list(remaining)
        return {"plan": steps}

    def router_after_judge(state: PlanExecuteState) -> str:
        if state.get("answer"):
            return "end"
        d = (state.get("_decision") or "").strip().lower()
        if d == "replan":
            return "replan"
        if d == "continue":
            return "execute" if (state.get("plan") or []) else "end"
        if d == "final":
            return "end"
        return "replan"

    g = StateGraph(PlanExecuteState)
    g.add_node("execute", executor_node)
    g.add_node("judge", judge_node)
    g.add_node("replan", replanner_node)

    g.set_entry_point("execute")
    g.add_edge("execute", "judge")
    g.add_conditional_edges("judge", router_after_judge, {"execute": "execute", "replan": "replan", "end": END})
    g.add_edge("replan", "execute")
    return g.compile()

