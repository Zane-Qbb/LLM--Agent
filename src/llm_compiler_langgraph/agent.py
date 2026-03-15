from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from rewoo_langgraph.tools import build_tools

from .prompts import CONTROLLER_SYSTEM, PLANNER_SYSTEM, REPLAN_SYSTEM, ControllerDecision, DagPlan, DagNode


class CycleDetectedError(Exception):
    """Raised when a cycle is detected in the DAG."""
    def __init__(self, cycle_nodes: List[str], message: str = "Cycle detected in DAG"):
        self.cycle_nodes = cycle_nodes
        self.message = message
        super().__init__(self.message)


class InvalidPlanError(Exception):
    """Raised when the plan validation fails."""
    pass


def _isolate_memory_by_generation(
    memory: Dict[str, str],
    generation: int,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Isolate memory by generation to prevent ID collisions.

    Returns:
        - namespaced_memory: Memory with generation prefix (e.g., "g1_n1")
        - reverse_map: Mapping from namespaced IDs to original IDs for debugging
    """
    namespaced: Dict[str, str] = {}
    reverse_map: Dict[str, str] = {}

    for key, value in memory.items():
        # Keys already have generation prefix
        namespaced[key] = value
        reverse_map[key] = key

    return namespaced, reverse_map


def _apply_generation_prefix(plan: DagPlan, generation: int) -> DagPlan:
    """
    Apply generation prefix to all node IDs, dependencies, AND arg placeholders.
    This ensures isolation between different outer iterations.

    Example: n1 -> g1_n1, deps: [n0] -> deps: [g1_n0], args "$n0" -> "$g1_n0"
    """
    prefix = f"g{generation}_"
    node_ids = {n.id for n in plan.nodes}

    def _prefix_placeholders(args: Dict[str, str]) -> Dict[str, str]:
        """Replace $nodeId placeholders with $g{gen}_nodeId in arg values."""
        new_args: Dict[str, str] = {}
        for k, v in args.items():
            if isinstance(v, str):
                def repl(m: re.Match[str]) -> str:
                    nid = m.group(1)
                    if nid in node_ids:
                        return f"${prefix}{nid}"
                    return m.group(0)
                new_args[k] = _PLACEHOLDER_RE.sub(repl, v)
            else:
                new_args[k] = v
        return new_args

    new_nodes = []
    for node in plan.nodes:
        new_node = DagNode(
            id=f"{prefix}{node.id}",
            description=node.description,
            tool=node.tool,
            args=_prefix_placeholders(node.args),
            deps=[f"{prefix}{d}" for d in node.deps],
        )
        new_nodes.append(new_node)

    return DagPlan(nodes=new_nodes)


def _strip_generation_prefix(key: str) -> Tuple[int, str]:
    """
    Strip generation prefix from a key.
    Returns (generation_number, original_key).
    Example: "g3_n1" -> (3, "n1")
    """
    match = re.match(r"g(\d+)_(.+)", key)
    if match:
        return int(match.group(1)), match.group(2)
    return 0, key


def _detect_cycle(nodes: List[DagNode]) -> Optional[List[str]]:
    """
    Detect cycles in the DAG using Kahn's algorithm.
    Returns a list of node IDs involved in the cycle, or None if no cycle exists.

    Time complexity: O(V + E) where V = number of nodes, E = number of edges.
    """
    node_ids = {n.id for n in nodes}

    # Build adjacency list and in-degree count
    adj: Dict[str, List[str]] = {nid: [] for nid in node_ids}
    indeg: Dict[str, int] = {nid: 0 for nid in node_ids}

    for n in nodes:
        for d in n.deps:
            if d in node_ids:  # Ignore deps that don't exist
                adj[d].append(n.id)
                indeg[n.id] += 1

    # Kahn's algorithm
    queue = [nid for nid, v in indeg.items() if v == 0]
    visited_count = 0

    while queue:
        nid = queue.pop(0)
        visited_count += 1
        for neighbor in adj[nid]:
            indeg[neighbor] -= 1
            if indeg[neighbor] == 0:
                queue.append(neighbor)

    # If we didn't visit all nodes, there's a cycle
    if visited_count != len(node_ids):
        # Find nodes that are part of the cycle (those with non-zero in-degree)
        cycle_nodes = [nid for nid, v in indeg.items() if v > 0]
        return cycle_nodes

    return None


def _validate_plan(plan: DagPlan) -> Tuple[bool, Optional[str]]:
    """
    Validate the plan structure and dependencies.
    Returns (is_valid, error_message).
    """
    node_ids = {n.id for n in plan.nodes}

    # Check for duplicate IDs
    if len(node_ids) != len(plan.nodes):
        return False, "Duplicate node IDs detected"

    # Check for self-dependencies
    for n in plan.nodes:
        if n.id in n.deps:
            return False, f"Node {n.id} depends on itself"

    # Check for invalid dependencies (deps that don't exist)
    for n in plan.nodes:
        for d in n.deps:
            if d not in node_ids:
                return False, f"Node {n.id} depends on non-existent node {d}"

    # Check for cycles
    cycle_nodes = _detect_cycle(plan.nodes)
    if cycle_nodes:
        return False, f"Cycle detected involving nodes: {cycle_nodes}"

    # Validate node structure: every node must have a tool
    for n in plan.nodes:
        if not n.tool:
            return False, f"Node {n.id} has no tool specified"

    return True, None


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
    """
    Best-effort extraction of the first JSON object from text.
    """
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"Model did not return JSON. Raw: {text[:500]}")
    return json.loads(m.group(0))


_PLACEHOLDER_RE = re.compile(r"\$(\w+)")


def _replace_placeholders(template: str, memory: Mapping[str, str]) -> str:
    """
    Replace $nodeId with corresponding memory value (string). Missing ids are left as debug markers.
    """

    def repl(m: re.Match[str]) -> str:
        nid = m.group(1)
        return memory.get(nid, f"<MISSING:{nid}>")

    return _PLACEHOLDER_RE.sub(repl, template)


def _render_args(raw_args: Mapping[str, Any], memory: Mapping[str, str]) -> Dict[str, Any]:
    rendered: Dict[str, Any] = {}
    for k, v in raw_args.items():
        if isinstance(v, str):
            rendered[k] = _replace_placeholders(v, memory)
        else:
            rendered[k] = v
    return rendered


_FAIL_PREFIXES = ("TOOL_ERROR(", "UNKNOWN_TOOL")


async def _is_node_failed(
    output: str,
    *,
    judge_llm: Optional[ChatOpenAI] = None,
) -> bool:
    """
    Hybrid failure detection.
    Fast path: check for known error prefixes produced by the executor.
    Fuzzy path: for ambiguous outputs, use a lightweight LLM call.
    """
    stripped = output.strip()
    if any(stripped.startswith(p) for p in _FAIL_PREFIXES):
        return True
    if not stripped or stripped == "None":
        return True

    if judge_llm is None:
        return False

    msg = [
        SystemMessage(content=(
            "You are a failure detector. Given a tool output, decide if the tool execution FAILED. "
            "Respond with ONLY 'yes' or 'no'.\n"
            "Examples of failure: HTTP errors (502, 404, timeout), empty results, error messages, "
            "'service unavailable', connection refused.\n"
            "Examples of success: any meaningful data, even partial."
        )),
        HumanMessage(content=f"Tool output:\n{stripped[:500]}"),
    ]
    try:
        out = await judge_llm.ainvoke(msg)
        answer = (out.content if isinstance(out.content, str) else "").strip().lower()
        return answer.startswith("yes")
    except Exception:
        return False


async def _plan_with_retry(
    *,
    planner_llm: ChatOpenAI,
    question: str,
    memory: Mapping[str, str],
    max_retries: int = 3,
) -> DagPlan:
    """
    Wrapper around _plan_once with self-correction retry mechanism.
    """
    last_error: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        try:
            return await _plan_once(
                planner_llm=planner_llm,
                question=question,
                memory=memory,
                retry_count=attempt - 1,
                prev_error=last_error,
            )
        except InvalidPlanError as e:
            last_error = str(e)
            print(f"[compiler.planner] 第 {attempt} 次尝试失败：{last_error}")

            if attempt == max_retries:
                print(f"[compiler.planner] 达到最大重试次数 ({max_retries})，使用简化 DAG 作为 fallback")
                # Fallback: create a minimal thought node that just summarizes
                return DagPlan(nodes=[
                    DagNode(
                        id="fallback_summary",
                        description="由于规划失败，直接对问题进行总结推理",
                        tool="join",
                        args={"prompt": "规划失败，请直接根据问题给出最佳答案"},
                        deps=[],
                    )
                ])

    # Should not reach here, but just in case
    return DagPlan(nodes=[
        DagNode(
            id="fallback_error",
            description="规划系统错误",
            tool="join",
            args={"prompt": "规划系统出现错误，请直接根据问题给出最佳答案"},
            deps=[],
        )
    ])


async def _plan_once(
    *,
    planner_llm: ChatOpenAI,
    question: str,
    memory: Mapping[str, str],
    retry_count: int = 0,
    prev_error: Optional[str] = None,
) -> DagPlan:
    """
    Call the Planner to get a DAG for the next outer iteration.
    With self-correction mechanism for invalid plans.
    """
    print("\n[compiler.planner] 开始规划 DAG...")
    if retry_count > 0:
        print(f"[compiler.planner] 重试次数：{retry_count}，上次错误：{prev_error}")
    start_time = time.time()

    # Build prompt with error feedback for self-correction
    error_feedback = ""
    if prev_error:
        error_feedback = f"\n\n【上次规划错误】{prev_error}\n请修正上述问题，重新生成有效的 DAG 计划。"

    payload = {
        "question": question,
        "memory": memory,
    }
    msg = [
        SystemMessage(content=PLANNER_SYSTEM + error_feedback),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2)),
    ]
    out = await planner_llm.ainvoke(msg)
    text = out.content if isinstance(out.content, str) else json.dumps(out.content, ensure_ascii=False)

    # Extract and validate JSON
    try:
        raw = _extract_json(text)
    except (ValueError, json.JSONDecodeError) as e:
        error_msg = f"JSON 解析失败：{e}. 原始输出：{text[:500]}"
        print(f"[compiler.planner] 错误：{error_msg}")
        raise InvalidPlanError(error_msg)

    # Validate plan structure
    try:
        plan = DagPlan.model_validate(raw)
    except Exception as e:
        error_msg = f"Pydantic 验证失败：{e}"
        print(f"[compiler.planner] 错误：{error_msg}")
        raise InvalidPlanError(error_msg)

    # Validate plan semantics (no cycles, valid deps, etc.)
    is_valid, error_msg = _validate_plan(plan)
    if not is_valid:
        print(f"[compiler.planner] 验证失败：{error_msg}")
        raise InvalidPlanError(error_msg)

    elapsed = time.time() - start_time
    print(f"[compiler.planner] 规划完成，耗时：{elapsed:.2f}s")
    print(f"[compiler.planner] DAG 节点数：{len(plan.nodes)}")

    # Print DAG structure and dependencies
    print("\n[compiler.planner] DAG 结构:")
    for node in plan.nodes:
        deps_str = f" <- [{', '.join(node.deps)}]" if node.deps else " (无依赖，可并行)"
        print(f"  {node.id}: [{node.tool}]{deps_str}")
        print(f"      描述：{node.description}")
        if node.args:
            print(f"      参数：{node.args}")

    return plan


def _collect_downstream(
    start_nid: str,
    adj: Dict[str, List[str]],
) -> Set[str]:
    """BFS to collect all transitive downstream nodes of start_nid (excluding start_nid itself)."""
    visited: Set[str] = set()
    queue = list(adj.get(start_nid, []))
    while queue:
        nid = queue.pop(0)
        if nid in visited:
            continue
        visited.add(nid)
        queue.extend(adj.get(nid, []))
    return visited


async def _local_replan(
    *,
    planner_llm: ChatOpenAI,
    question: str,
    failed_node: DagNode,
    failed_output: str,
    frozen_downstream: List[DagNode],
    memory: Mapping[str, str],
) -> DagPlan:
    """
    Call the Planner to generate a replacement sub-DAG for a failed subtree.
    Returns a DagPlan with new nodes that replace the failed node + its downstream.
    """
    print(f"\n[compiler.replan] 开始局部重规划，失败节点：{failed_node.id}")
    start_time = time.time()

    payload = {
        "question": question,
        "failed_node": {
            "id": failed_node.id,
            "description": failed_node.description,
            "tool": failed_node.tool,
            "args": failed_node.args,
            "error_output": failed_output[:500],
        },
        "frozen_downstream": [
            {"id": n.id, "description": n.description, "tool": n.tool, "args": n.args}
            for n in frozen_downstream
        ],
        "available_memory": {k: v[:200] for k, v in memory.items()},
    }
    msg = [
        SystemMessage(content=REPLAN_SYSTEM),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2)),
    ]
    out = await planner_llm.ainvoke(msg)
    text = out.content if isinstance(out.content, str) else json.dumps(out.content, ensure_ascii=False)

    raw = _extract_json(text)
    plan = DagPlan.model_validate(raw)

    is_valid, error_msg = _validate_plan(plan)
    if not is_valid:
        raise InvalidPlanError(f"局部重规划验证失败：{error_msg}")

    elapsed = time.time() - start_time
    print(f"[compiler.replan] 局部重规划完成，耗时：{elapsed:.2f}s，新节点数：{len(plan.nodes)}")
    for node in plan.nodes:
        deps_str = f" <- [{', '.join(node.deps)}]" if node.deps else " (无依赖)"
        print(f"  {node.id}: [{node.tool}]{deps_str}")
        print(f"      描述：{node.description}")

    return plan


def _apply_replan_prefix(
    plan: DagPlan,
    prefix: str,
    existing_memory_keys: Set[str],
) -> DagPlan:
    """
    Apply replan namespace prefix to sub-DAG node IDs, deps, and arg placeholders.
    Placeholders that reference existing memory keys are left untouched.
    """
    sub_node_ids = {n.id for n in plan.nodes}

    def _prefix_placeholders(args: Dict[str, str]) -> Dict[str, str]:
        new_args: Dict[str, str] = {}
        for k, v in args.items():
            if isinstance(v, str):
                def repl(m: re.Match[str]) -> str:
                    nid = m.group(1)
                    if nid in sub_node_ids:
                        return f"${prefix}{nid}"
                    if nid in existing_memory_keys:
                        return m.group(0)
                    return m.group(0)
                new_args[k] = _PLACEHOLDER_RE.sub(repl, v)
            else:
                new_args[k] = v
        return new_args

    new_nodes = []
    for node in plan.nodes:
        new_deps = []
        for d in node.deps:
            if d in sub_node_ids:
                new_deps.append(f"{prefix}{d}")
            elif d in existing_memory_keys:
                new_deps.append(d)
            else:
                new_deps.append(f"{prefix}{d}")
        new_node = DagNode(
            id=f"{prefix}{node.id}",
            description=node.description,
            tool=node.tool,
            args=_prefix_placeholders(node.args),
            deps=new_deps,
        )
        new_nodes.append(new_node)

    return DagPlan(nodes=new_nodes)


def _hot_merge_subdag(
    *,
    sub_plan: DagPlan,
    failed_nid: str,
    frozen_nids: Set[str],
    nodes: Dict[str, DagNode],
    adj: Dict[str, List[str]],
    indeg: Dict[str, int],
    mem: Dict[str, str],
) -> List[str]:
    """
    Hot-merge a replan sub-DAG into the live DAG.
    - Remove frozen nodes from the main DAG structures
    - Inject new sub-DAG nodes
    - Rewire: any node that depended on a frozen node now depends on sub-DAG leaf nodes
    - Returns list of new ready node IDs (indeg == 0 in the sub-DAG)

    The failed_nid itself is already in mem (with error output), we remove it to clean up.
    """
    # Identify the join node(s) — nodes that depended on frozen nodes but are NOT themselves frozen
    join_candidates: Set[str] = set()
    for frozen_nid in frozen_nids:
        for downstream in adj.get(frozen_nid, []):
            if downstream not in frozen_nids:
                join_candidates.add(downstream)
    # Also check the failed node's direct downstream that wasn't frozen
    for downstream in adj.get(failed_nid, []):
        if downstream not in frozen_nids:
            join_candidates.add(downstream)

    print(f"[compiler.replan.merge] 移除冻结节点：{frozen_nids}")
    print(f"[compiler.replan.merge] 需要重连的下游节点（join 等）：{join_candidates}")

    # Remove failed node and frozen nodes from main DAG structures
    all_removed = frozen_nids | {failed_nid}
    for nid in all_removed:
        nodes.pop(nid, None)
        adj.pop(nid, None)
        indeg.pop(nid, None)
        mem.pop(nid, None)
    # Clean up adjacency references to removed nodes
    for nid in list(adj.keys()):
        adj[nid] = [nb for nb in adj[nid] if nb not in all_removed]

    # Inject sub-DAG nodes
    sub_node_ids = set()
    for n in sub_plan.nodes:
        nodes[n.id] = n
        adj[n.id] = []
        indeg[n.id] = 0
        sub_node_ids.add(n.id)

    # Build sub-DAG internal edges
    for n in sub_plan.nodes:
        for d in n.deps:
            if d in sub_node_ids:
                adj[d].append(n.id)
                indeg[n.id] += 1
            elif d in nodes:
                # Dep on an existing (successful) node in the main DAG
                adj.setdefault(d, []).append(n.id)
                indeg[n.id] += 1

    # Find leaf nodes of the sub-DAG (nodes with no downstream within sub-DAG)
    sub_leaves: Set[str] = set(sub_node_ids)
    for n in sub_plan.nodes:
        for d in n.deps:
            if d in sub_node_ids:
                sub_leaves.discard(d)
    # Actually, leaves = nodes that have no outgoing edges within the sub-DAG
    sub_leaves = set()
    for nid in sub_node_ids:
        has_sub_downstream = any(nb in sub_node_ids for nb in adj.get(nid, []))
        if not has_sub_downstream:
            sub_leaves.add(nid)

    print(f"[compiler.replan.merge] 子 DAG 叶节点：{sub_leaves}")

    # Rewire join candidates: replace deps on removed nodes with deps on sub-DAG leaves
    for join_nid in join_candidates:
        if join_nid not in nodes:
            continue
        join_node = nodes[join_nid]
        old_deps = join_node.deps
        new_deps = [d for d in old_deps if d not in all_removed]
        new_deps.extend(sub_leaves)
        # Deduplicate
        new_deps = list(dict.fromkeys(new_deps))
        # Update the node (create a new DagNode since it's a Pydantic model)
        nodes[join_nid] = DagNode(
            id=join_nid,
            description=join_node.description,
            tool=join_node.tool,
            args=join_node.args,
            deps=new_deps,
        )
        # Recalculate indeg for the join node
        indeg[join_nid] = sum(1 for d in new_deps if d in nodes)
        # Add edges from sub-DAG leaves to the join node
        for leaf_nid in sub_leaves:
            if join_nid not in adj.get(leaf_nid, []):
                adj.setdefault(leaf_nid, []).append(join_nid)

        print(f"[compiler.replan.merge] 重连 {join_nid}: deps {old_deps} -> {new_deps}")

    # Return new ready nodes (indeg == 0 within the sub-DAG)
    new_ready = [nid for nid in sub_node_ids if indeg.get(nid, 0) == 0]
    print(f"[compiler.replan.merge] 新就绪节点：{new_ready}")
    return new_ready


async def _execute_dag_once(
    *,
    question: str,
    plan: DagPlan,
    tool_map: Mapping[str, Any],
    memory: Dict[str, str],
    planner_llm: Optional[ChatOpenAI] = None,
    judge_llm: Optional[ChatOpenAI] = None,
    max_local_replans: int = 2,
) -> Dict[str, str]:
    """
    Dynamic DAG executor with local fault tolerance and partial replan:
    - Topological batch execution with parallel ready-queue
    - On node failure: freeze downstream subtree, trigger local replan
    - Hot-merge replacement sub-DAG into the live execution
    - Successful parallel branches continue uninterrupted
    """
    nodes: Dict[str, DagNode] = {n.id: n for n in plan.nodes}
    adj: Dict[str, List[str]] = {nid: [] for nid in nodes}
    indeg: Dict[str, int] = {nid: 0 for nid in nodes}
    for n in plan.nodes:
        for d in n.deps:
            if d not in nodes:
                continue
            adj[d].append(n.id)
            indeg[n.id] += 1

    print("\n[compiler.executor] DAG 依赖关系图:")
    print(f"  节点总数：{len(nodes)}")
    print(f"  初始可并行节点 (入度=0): {[nid for nid, v in indeg.items() if v == 0]}")

    ready: List[str] = [nid for nid, v in indeg.items() if v == 0]
    mem: Dict[str, str] = dict(memory)
    frozen_nodes: Set[str] = set()
    local_replan_count = 0
    replan_tasks: List[asyncio.Task] = []
    sub_gen_counter = 0

    print(f"\n[compiler.executor] 开始执行 DAG，初始 memory 键：{list(memory.keys())}")

    node_start_times: Dict[str, float] = {}

    async def run_single_node(nid: str) -> Tuple[str, str]:
        node = nodes[nid]
        node_start_times[nid] = time.time()

        tool = tool_map.get(node.tool)
        if not tool:
            print(f"\n[compiler.executor] UNKNOWN_TOOL for node={nid}: {node.tool}")
            return nid, f"UNKNOWN_TOOL: {node.tool}"

        rendered_args = _render_args(node.args, mem)
        print(f"\n[compiler.executor] 执行节点：{nid}")
        print(f"  工具：{node.tool}")
        print(f"  依赖：{node.deps}")
        print(f"  渲染后参数：{rendered_args}")
        try:
            if len(rendered_args) == 1 and isinstance(next(iter(rendered_args.values())), str):
                arg_val = next(iter(rendered_args.values()))
                result = await tool.ainvoke(arg_val)
            else:
                result = await tool.ainvoke(rendered_args)
        except Exception as e:
            print(f"[compiler.executor] node={nid} tool={node.tool} ERROR: {e}")
            return nid, f"TOOL_ERROR({node.tool}): {e}"
        return nid, str(result)

    batch_num = 0
    while ready or replan_tasks:
        # If no ready nodes but replan tasks are pending, wait for them
        if not ready and replan_tasks:
            print(f"\n[compiler.executor] 等待 {len(replan_tasks)} 个局部重规划完成...")
            done_tasks, _ = await asyncio.wait(replan_tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done_tasks:
                replan_tasks.remove(task)
                try:
                    new_ready = task.result()
                    ready.extend(new_ready)
                except Exception as e:
                    print(f"[compiler.executor] 局部重规划失败：{e}，降级处理")
            continue

        if not ready:
            break

        batch = list(ready)
        ready.clear()
        batch_num += 1

        print(f"\n[compiler.executor] >>> 第 {batch_num} 批并行执行，节点：{batch}")
        batch_start = time.time()

        coros = [run_single_node(nid) for nid in batch]
        results = await asyncio.gather(*coros, return_exceptions=True)

        batch_elapsed = time.time() - batch_start
        print(f"[compiler.executor] <<< 第 {batch_num} 批完成，总耗时：{batch_elapsed:.2f}s")

        for item in results:
            if isinstance(item, Exception):
                continue
            nid, out_str = item

            node_elapsed = time.time() - node_start_times.get(nid, 0)
            print(f"\n[compiler.executor] 节点 {nid} 执行完成，耗时：{node_elapsed:.2f}s")
            print(f"  输出：{out_str[:150]}{'...' if len(out_str) > 150 else ''}")

            node_obj = nodes.get(nid)
            is_join = node_obj and node_obj.tool == "join"
            failed = False
            if not is_join:
                failed = await _is_node_failed(out_str, judge_llm=judge_llm)

            if failed and planner_llm and local_replan_count < max_local_replans:
                print(f"\n[compiler.executor] *** 节点 {nid} 执行失败，触发局部重规划 ***")
                local_replan_count += 1
                sub_gen_counter += 1

                downstream = _collect_downstream(nid, adj)
                downstream_not_join = {d for d in downstream if nodes.get(d) and nodes[d].tool != "join"}
                frozen_nodes.update(downstream_not_join)

                print(f"  冻结下游节点：{downstream_not_join}")
                print(f"  保留的 join 节点：{downstream - downstream_not_join}")

                failed_node_obj = nodes[nid]
                frozen_node_objs = [nodes[d] for d in downstream_not_join if d in nodes]

                # Build replan prefix from the outer generation prefix of the failed node
                gen_match = re.match(r"(g\d+_)", nid)
                gen_prefix = gen_match.group(1) if gen_match else ""
                replan_prefix = f"{gen_prefix}r{sub_gen_counter}_"

                async def do_replan(
                    _failed_node=failed_node_obj,
                    _failed_output=out_str,
                    _frozen_objs=frozen_node_objs,
                    _frozen_nids=downstream_not_join,
                    _replan_prefix=replan_prefix,
                    _failed_nid=nid,
                ) -> List[str]:
                    try:
                        sub_plan = await _local_replan(
                            planner_llm=planner_llm,
                            question=question,
                            failed_node=_failed_node,
                            failed_output=_failed_output,
                            frozen_downstream=_frozen_objs,
                            memory=mem,
                        )
                        sub_plan = _apply_replan_prefix(
                            sub_plan, _replan_prefix, set(mem.keys()),
                        )
                        print(f"\n[compiler.replan] 应用命名空间 '{_replan_prefix}' 后节点:")
                        for n in sub_plan.nodes:
                            print(f"  {n.id} (tool={n.tool}, deps={n.deps})")

                        new_ready = _hot_merge_subdag(
                            sub_plan=sub_plan,
                            failed_nid=_failed_nid,
                            frozen_nids=_frozen_nids,
                            nodes=nodes,
                            adj=adj,
                            indeg=indeg,
                            mem=mem,
                        )
                        return new_ready
                    except Exception as e:
                        print(f"[compiler.replan] 局部重规划异常：{e}，降级为写入错误到 memory")
                        mem[_failed_nid] = _failed_output
                        for fn in _frozen_nids:
                            frozen_nodes.discard(fn)
                        # Unfreeze: release downstream as if success (with error data)
                        for nb in adj.get(_failed_nid, []):
                            if nb in indeg:
                                indeg[nb] -= 1
                                if indeg[nb] == 0:
                                    ready.append(nb)
                        return []

                task = asyncio.create_task(do_replan())
                replan_tasks.append(task)

            elif failed:
                # Exceeded max local replans or no planner — degrade
                print(f"[compiler.executor] 节点 {nid} 失败，已超过最大局部重规划次数，降级处理")
                mem[nid] = out_str
                for nb in adj.get(nid, []):
                    indeg[nb] -= 1
                    if indeg[nb] == 0 and nb not in frozen_nodes:
                        ready.append(nb)
            else:
                # Success path
                mem[nid] = out_str
                print(f"  [compiler.executor] Memory 更新：{nid} = {out_str[:50]}{'...' if len(out_str) > 50 else ''}")

                for nb in adj.get(nid, []):
                    indeg[nb] -= 1
                    if indeg[nb] == 0 and nb not in frozen_nodes:
                        print(f"  [compiler.executor] 依赖满足，节点 {nb} 加入就绪队列")
                        ready.append(nb)

    print(f"\n[compiler.executor] DAG 执行完成，共 {batch_num} 个并行批次，局部重规划 {local_replan_count} 次")
    print(f"[compiler.executor] 最终 memory 键：{list(mem.keys())}")
    return mem


async def _controller_decide(
    *,
    controller_llm: ChatOpenAI,
    question: str,
    memory: Mapping[str, str],
) -> ControllerDecision:
    print("\n[compiler.controller] 开始决策是否继续迭代...")
    start_time = time.time()

    payload = {
        "question": question,
        "memory": memory,
    }
    msg = [
        SystemMessage(content=CONTROLLER_SYSTEM),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2)),
    ]
    out = await controller_llm.ainvoke(msg)
    text = out.content if isinstance(out.content, str) else json.dumps(out.content, ensure_ascii=False)
    raw = _extract_json(text)
    decision = ControllerDecision.model_validate(raw)

    elapsed = time.time() - start_time
    print(f"[compiler.controller] 决策完成，耗时：{elapsed:.2f}s")
    print(f"[compiler.controller] action={decision.action}, reason={decision.reason}")
    return decision


def _build_llm_tools(
    reasoning_llm: ChatOpenAI,
    question: str,
) -> List[Any]:
    """Build LLM-backed tools: llm_reasoning and join."""
    from langchain_core.tools import tool

    @tool("llm_reasoning")
    async def llm_reasoning(prompt: str) -> str:
        """Use LLM to analyze, reason, summarize, or answer questions based on the given prompt."""
        msg = [
            SystemMessage(content=(
                "You are a reasoning engine inside an LLM Compiler DAG. "
                "Perform the task described in the user prompt. "
                "Be concise and return your result in Chinese."
            )),
            HumanMessage(content=f"用户原始问题：{question}\n\n当前任务：{prompt}"),
        ]
        out = await reasoning_llm.ainvoke(msg)
        return out.content if isinstance(out.content, str) else json.dumps(out.content, ensure_ascii=False)

    @tool("join")
    async def join(prompt: str) -> str:
        """Aggregate results from previous nodes into a final answer. The prompt should describe how to combine results."""
        msg = [
            SystemMessage(content=(
                "You are the final aggregation step of an LLM Compiler DAG. "
                "Combine and synthesize the provided information into a clear, complete answer. "
                "Answer in Chinese."
            )),
            HumanMessage(content=f"用户原始问题：{question}\n\n聚合指令：{prompt}"),
        ]
        out = await reasoning_llm.ainvoke(msg)
        return out.content if isinstance(out.content, str) else json.dumps(out.content, ensure_ascii=False)

    return [llm_reasoning, join]


async def run_compiler(
    question: str,
    *,
    workspace_root: Optional[str] = None,
    max_outer_loops: int = 4,
) -> Dict[str, Any]:
    """
    High-level entry point for the LLM Compiler paradigm.

    Implements the standard LLM Compiler architecture:
    - Function Calling Planner (natural language -> DAG of tool calls with placeholders)
    - Task Fetching Unit (topological execution with parallel ready-queue batches)
    - Executor (all nodes are tool calls, including llm_reasoning and join)
    - Dynamic Replan: outer loop re-invokes Planner using updated memory
    - Generation Isolation: node IDs prefixed with iteration number to prevent collisions
    - Cycle Detection: Kahn's algorithm detects circular dependencies
    - Self-Correction: Invalid plans trigger retry with error feedback

    Returns a dict including:
    - "answer": final answer string (or best-effort)
    - "memory": all node outputs across iterations (with generation prefixes)
    - "iterations": number of outer iterations actually run
    """
    total_start = time.time()

    workspace_root = workspace_root or os.getcwd()
    base_tools = build_tools(workspace_root)

    planner_llm = _get_llm("LLM_COMPILER_PLANNER_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    reasoning_llm = _get_llm("LLM_COMPILER_THOUGHT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    controller_llm = _get_llm("LLM_COMPILER_CONTROLLER_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))

    llm_tools = _build_llm_tools(reasoning_llm, question)
    all_tools = base_tools + llm_tools
    tool_map = {t.name: t for t in all_tools}

    print(f"[LLM Compiler] 初始化完成")
    print(f"  工作目录：{workspace_root}")
    print(f"  可用工具：{list(tool_map.keys())}")
    print(f"  最大迭代次数：{max_outer_loops}")
    print(f"  问题：{question}")
    print(f"  鲁棒性特性：环路检测 | 自我修正 | 命名空间隔离 | 局部容错重规划")

    memory: Dict[str, str] = {}
    iterations = 0

    for outer in range(1, max_outer_loops + 1):
        iteration_start = time.time()
        iterations = outer
        print(f"\n{'='*60}")
        print(f"  [LLM Compiler] 开始第 {outer} 轮迭代 (共 {max_outer_loops} 轮)")
        print(f"{'='*60}")

        print(f"\n[compiler] 当前 memory 状态（所有历史代际）：{list(memory.keys())}")

        plan = await _plan_with_retry(
            planner_llm=planner_llm,
            question=question,
            memory=memory,
            max_retries=3,
        )

        plan = _apply_generation_prefix(plan, generation=outer)
        print(f"\n[compiler] 应用命名空间隔离后节点 IDs:")
        for node in plan.nodes:
            print(f"  {node.id} (tool={node.tool}, deps={node.deps})")

        memory = await _execute_dag_once(
            question=question,
            plan=plan,
            tool_map=tool_map,
            memory=memory,
            planner_llm=planner_llm,
            judge_llm=reasoning_llm,
            max_local_replans=2,
        )
        decision = await _controller_decide(controller_llm=controller_llm, question=question, memory=memory)

        iteration_elapsed = time.time() - iteration_start
        print(f"\n[LLM Compiler] 第 {outer} 轮迭代完成，耗时：{iteration_elapsed:.2f}s")

        if decision.action == "final":
            answer = (decision.answer or "").strip()
            if not answer:
                answer = "控制器选择了 final，但没有提供答案。请根据上文 memory 中的信息自行总结。"

            total_elapsed = time.time() - total_start
            print(f"\n{'='*60}")
            print(f"  [LLM Compiler] 任务完成!")
            print(f"  总耗时：{total_elapsed:.2f}s")
            print(f"  总迭代轮数：{iterations}")
            print(f"  Memory 键总数：{len(memory)}")
            print(f"{'='*60}")

            return {
                "answer": answer,
                "memory": memory,
                "iterations": iterations,
            }

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  [LLM Compiler] 达到最大迭代次数，强制结束")
    print(f"  总耗时：{total_elapsed:.2f}s")
    print(f"{'='*60}")

    return {
        "answer": "在预设的最大重规划轮数内未完全解决问题，以下是基于当前中间结果的最佳总结。",
        "memory": memory,
        "iterations": iterations,
    }

