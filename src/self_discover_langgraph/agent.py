from __future__ import annotations

import os
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from .prompts import REASONING_MODULES


class SelfDiscoverState(TypedDict, total=False):
    task_description: str
    selected_modules: str
    adapted_modules: str
    reasoning_structure: str
    solution: str


def _get_llm() -> ChatOpenAI:
    model = os.getenv("OPENAI_MODEL", "gpt-4-0125-preview")
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set env vars OPENAI_API_KEY (and optionally OPENAI_BASE_URL/OPENAI_MODEL)."
        )
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.1,
        max_tokens=2048,
    )


def select_reasoning_modules(state: SelfDiscoverState) -> SelfDiscoverState:
    """Step 1: SELECT relevant reasoning modules for the task."""
    task_description = state["task_description"]
    prompt = (
        f"Given the task: {task_description}, which of the following reasoning modules are relevant? "
        f"Do not elaborate on why.\n\n" + "\n".join(REASONING_MODULES)
    )
    llm = _get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"selected_modules": response.content.strip()}


def adapt_reasoning_modules(state: SelfDiscoverState) -> SelfDiscoverState:
    """Step 2: ADAPT the selected reasoning modules to be more specific to the task."""
    selected_modules = state["selected_modules"]
    task_description = state["task_description"]
    prompt = (
        f"Without working out the full solution, adapt the following reasoning modules to be specific to our task:\n"
        f"{selected_modules}\n\nOur task:\n{task_description}"
    )
    llm = _get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"adapted_modules": response.content.strip()}


def implement_reasoning_structure(state: SelfDiscoverState) -> SelfDiscoverState:
    """Step 3: IMPLEMENT the adapted reasoning modules into an actionable reasoning structure."""
    adapted_modules = state["adapted_modules"]
    task_description = state["task_description"]
    prompt = (
        f"Without working out the full solution, create an actionable reasoning structure for the task using these adapted reasoning modules:\n"
        f"{adapted_modules}\n\nTask Description:\n{task_description}"
    )
    llm = _get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"reasoning_structure": response.content.strip()}


def execute_reasoning_structure(state: SelfDiscoverState) -> SelfDiscoverState:
    """Execute the reasoning structure to solve a specific task instance."""
    reasoning_structure = state["reasoning_structure"]
    task_description = state["task_description"]
    prompt = (
        f"Using the following reasoning structure: {reasoning_structure}\n\n"
        f"Solve this task, providing your final answer: {task_description}"
    )
    llm = _get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"solution": response.content.strip()}


def build_graph() -> StateGraph:
    workflow = StateGraph(SelfDiscoverState)

    workflow.add_node("select", select_reasoning_modules)
    workflow.add_node("adapt", adapt_reasoning_modules)
    workflow.add_node("implement", implement_reasoning_structure)
    workflow.add_node("execute", execute_reasoning_structure)

    workflow.set_entry_point("select")
    workflow.add_edge("select", "adapt")
    workflow.add_edge("adapt", "implement")
    workflow.add_edge("implement", "execute")
    workflow.add_edge("execute", END)

    return workflow.compile()

__all__ = ["build_graph", "_get_llm"]
