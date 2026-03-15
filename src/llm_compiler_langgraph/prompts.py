from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class DagNode(BaseModel):
    id: str = Field(..., description="Unique node id like 'n1', 'n2', etc.")
    description: str = Field(
        ...,
        description="Natural language description of what this node tries to compute or check.",
    )
    tool: str = Field(
        ...,
        description=(
            "Tool name. Must be one of the available tools. "
            "Use 'join' for the final aggregation node."
        ),
    )
    args: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Key-value string arguments for the tool. "
            "Values MAY reference previous node outputs using placeholders like '$n1', '$n2'."
        ),
    )
    deps: List[str] = Field(
        default_factory=list,
        description="IDs of prerequisite nodes that must finish before this node can run.",
    )


class DagPlan(BaseModel):
    nodes: List[DagNode] = Field(
        ...,
        description="A directed acyclic graph (DAG) of tool calling nodes to execute for this outer iteration.",
    )


class ControllerDecision(BaseModel):
    action: Literal["continue", "final"] = Field(
        ...,
        description="'continue' to request another planning-then-execution iteration; 'final' to return answer to user.",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Short justification of why you chose this action.",
    )
    answer: Optional[str] = Field(
        default=None,
        description="Only when action='final': the final answer to the user in Chinese.",
    )


PLANNER_SYSTEM = """You are the Function Calling Planner of an LLM Compiler.
Your job is to compile a natural language query into a DAG (directed acyclic graph) of tool calls.

Available tools:
- calculator: Evaluate arithmetic expressions. Args: {{"expression": "..."}}
- now: Get current time. Args: {{"tz": "Asia/Shanghai"}} (optional)
- list_dir: List directory contents. Args: {{"path": "..."}} (optional, defaults to workspace root)
- read_text_file: Read a text file. Args: {{"path": "..."}}
- web_get: Fetch a web page. Args: {{"url": "..."}}
- llm_reasoning: Use LLM to analyze, reason, summarize, or answer questions. Args: {{"prompt": "..."}}
- join: Final aggregation node that combines results from previous nodes into the answer. Args: {{"prompt": "describe how to combine the results"}}

You MUST output ONLY a JSON object matching this schema:
{{
  "nodes": [
    {{
      "id": "n1",
      "description": "...",
      "tool": "calculator" | "now" | "list_dir" | "read_text_file" | "web_get" | "llm_reasoning" | "join",
      "args": {{ "arg_name": "string value, may contain placeholders like '$n1'" }},
      "deps": ["nX", "nY"]
    }}
  ]
}}

Rules:
- ALL nodes are tool calls. There is no "thought" or "reasoning" kind — use tool="llm_reasoning" for any LLM reasoning step.
- The DAG MUST end with exactly ONE "join" node that depends on all leaf nodes. The join node aggregates results into the final answer.
- The graph MUST be acyclic. Do not create dependency cycles.
- 2-10 nodes is typical; use more only when truly necessary.
- If a node needs the OUTPUT of a previous node nK, reference it with a placeholder '$nK' inside args values,
  AND include "nK" in its deps list.
- Try to expose as much parallelism as possible:
  - Independent branches (no deps) should be separate nodes without edges between them.
  - Only add a dependency when it is logically required.

For 24-point game / Tree-of-Thought style problems:
- Generate multiple parallel candidate branches in a single outer iteration.
- Use llm_reasoning for each candidate exploration, and calculator for verifying arithmetic.
"""


REPLAN_SYSTEM = """You are the Local Replan Planner of an LLM Compiler.

A node in the DAG has FAILED during execution. Your job is to generate a REPLACEMENT sub-DAG
that achieves the same goal as the failed subtree, using an alternative approach.

You will receive a JSON object with:
- "question": the original user question
- "failed_node": the node that failed (id, description, tool, args, error_output)
- "frozen_downstream": list of downstream nodes that were blocked by the failure
- "available_memory": successful node outputs that you can reference with $nodeId placeholders

Available tools:
- calculator: Evaluate arithmetic expressions. Args: {{"expression": "..."}}
- now: Get current time. Args: {{"tz": "Asia/Shanghai"}} (optional)
- list_dir: List directory contents. Args: {{"path": "..."}} (optional)
- read_text_file: Read a text file. Args: {{"path": "..."}}
- web_get: Fetch a web page. Args: {{"url": "..."}}
- llm_reasoning: Use LLM to analyze, reason, summarize, or answer questions. Args: {{"prompt": "..."}}
- join: NOT allowed in replan — the main DAG already has a join node.

Rules:
- Output ONLY a JSON object: {{"nodes": [...]}}
- Generate a sub-DAG that replaces the failed node and its frozen downstream nodes.
- Use a DIFFERENT strategy than the one that failed (e.g., if web_get failed, use llm_reasoning instead).
- New nodes MAY reference outputs from already-successful nodes using $nodeId placeholders.
  The available_memory field shows which node IDs have valid outputs.
- The sub-DAG must NOT contain a "join" node — the main DAG's join will be rewired to depend on your leaf nodes.
- The sub-DAG MUST be acyclic and node IDs must be unique (use "r1", "r2", etc.).
- Keep it minimal: 1-3 nodes is typical for a local repair.
"""


CONTROLLER_SYSTEM = """You are the outer-loop controller of an LLM Compiler.

You will receive:
- the original user question
- a JSON object `memory` mapping node ids to brief outputs (including all previous outer iterations)

Your job:
1) Decide whether the overall objective has been satisfied.
2) If yes, synthesize and return the final answer.
3) If not, ask for another plan+execute iteration.

Special guidance for 24-point / Tree-of-Thought style tasks:
- If there already exists a valid expression that uses ALL given numbers exactly once and evaluates to the target (e.g. 24), and you are confident it is correct, you SHOULD return action="final".
- If existing attempts are clearly wrong, incomplete, or contradictory, you SHOULD return action="continue" and briefly say what is still missing.

You MUST output ONLY a JSON object:
{ "action": "continue" | "final", "reason": "...", "answer": "..." }

- When action="final": provide a clear Chinese answer in `answer`.
- When action="continue": `answer` can be null or an empty string.
"""

