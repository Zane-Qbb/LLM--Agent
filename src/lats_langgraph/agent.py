"""LATS (Language Agent Tree Search) — 基于 LangGraph 的六步 MCTS 循环实现。

Select → Expand → Evaluate → Simulate → Backpropagate → Reflect
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from .prompts import (
    CRITIC_SYSTEM,
    FINAL_ANSWER_SYSTEM,
    GENERATOR_SYSTEM,
    REFLECT_SYSTEM,
)
from .tools import ToolExecutor

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


class TreeNode:
    """MCTS 搜索树节点。"""

    def __init__(self, state_text: str, parent: Optional["TreeNode"] = None):
        self.state_text = state_text
        self.parent = parent
        self.children: List[TreeNode] = []
        self.visits: int = 0
        self.value: float = 0.0          # backprop 累积值（由 backprop 维护）
        self.reward: float = 0.0         # simulate 产生的即时奖励
        self.heuristic_value: float = 0.0  # Critic 评分（用于 Evaluate 阶段选最优候选）
        self.is_terminal: bool = False
        self.is_success: bool = False
        self.action: str = ""
        self.action_type: str = ""
        self.tool_name: Optional[str] = None
        self.tool_input: Optional[str] = None
        self.reasoning: Optional[str] = None
        self.depth: int = (parent.depth + 1) if parent else 0

    def uct(self, c: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.value / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def __repr__(self) -> str:
        avg = self.value / self.visits if self.visits > 0 else 0.0
        return (
            f"TreeNode(depth={self.depth}, visits={self.visits}, "
            f"value={self.value:.2f}, avg={avg:.2f}, reward={self.reward:.2f}, "
            f"action={self.action[:40]!r})"
        )


class LATSState(TypedDict, total=False):
    question: str
    root: Any  # TreeNode (LangGraph 不直接序列化自定义类，用 Any)
    current_node: Any
    candidates: List[Any]
    reflections: List[str]
    max_budget: int
    current_step: int
    best_node: Any
    answer: str


# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------

N_CANDIDATES = 3
MAX_DEPTH = 6
DEFAULT_BUDGET = 8


def _get_llm(temperature: float = 0.7) -> ChatOpenAI:
    model = os.getenv("OPENAI_MODEL", "gpt-4-0125-preview")
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set env vars OPENAI_API_KEY "
            "(and optionally OPENAI_BASE_URL / OPENAI_MODEL)."
        )
    kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": 2048,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def _parse_json_array(text: str) -> List[Dict]:
    """从 LLM 输出中提取第一个 JSON 数组。"""
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        raise ValueError(f"No JSON array found in LLM output: {text[:300]}")
    return json.loads(m.group(0))


def _get_trajectory(node: TreeNode) -> str:
    """从当前节点回溯到根，拼出完整轨迹。"""
    parts: List[str] = []
    cur = node
    while cur is not None:
        if cur.action:
            parts.append(f"[Depth {cur.depth}] Action: {cur.action}")
        if cur.state_text and cur.parent is not None:
            last_obs = cur.state_text.split("\nObservation:")
            if len(last_obs) > 1:
                parts.append(f"  Observation: {last_obs[-1].strip()[:500]}")
        cur = cur.parent
    parts.reverse()
    return "\n".join(parts) if parts else "(empty trajectory)"


# ---------------------------------------------------------------------------
# LangGraph 节点实现
# ---------------------------------------------------------------------------

_tool_executor: Optional[ToolExecutor] = None


def _get_tool_executor() -> ToolExecutor:
    global _tool_executor
    if _tool_executor is None:
        _tool_executor = ToolExecutor(workspace_root=os.getcwd())
    return _tool_executor


def node_select(state: LATSState) -> dict:
    """1. Select — UCT 选择最值得探索的叶节点。"""
    node: TreeNode = state["root"]
    while node.children and not node.is_terminal:
        node = max(node.children, key=lambda c: c.uct())
    print(f"\n[select] 选中节点: depth={node.depth}, visits={node.visits}, value={node.value:.2f}")
    return {"current_node": node}


def node_expand(state: LATSState) -> dict:
    """2. Expand — LLM Generator 结合反思记忆，在选中节点上生成 K 个候选动作。"""
    node: TreeNode = state["current_node"]
    if node.is_terminal:
        print("[expand] 节点已终止，跳过扩展")
        return {"candidates": []}

    if node.depth >= MAX_DEPTH:
        print(f"[expand] 达到最大深度 {MAX_DEPTH}，标记为终止")
        node.is_terminal = True
        return {"candidates": []}

    # 该节点已由之前的 expand 创建（有 action），但还没被 simulate（visits==0）
    # 直接把自己作为唯一候选推入 evaluate→simulate，无需再展开子节点
    if node.action and node.visits == 0:
        print(f"[expand] 节点已有 action 且未 simulate，直接作为候选推进")
        return {"candidates": [node]}

    if node.children:
        print(f"[expand] 节点已有 {len(node.children)} 个子节点，跳过重复扩展")
        return {"candidates": node.children}

    te = _get_tool_executor()
    reflections_text = "\n".join(state.get("reflections", [])) or "（无）"
    prompt = GENERATOR_SYSTEM.format(
        tool_descriptions=te.tool_descriptions,
        reflections=reflections_text,
        state_text=node.state_text,
        n_candidates=N_CANDIDATES,
    )

    llm = _get_llm(temperature=0.8)
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content if isinstance(response.content, str) else json.dumps(response.content)

    try:
        actions = _parse_json_array(raw)
    except Exception as e:
        print(f"[expand] 解析候选动作失败: {e}，使用 fallback")
        actions = [{"thought": "直接推理", "action_type": "reasoning", "reasoning": "让我重新分析问题"}]

    candidates: List[TreeNode] = []
    for act in actions[:N_CANDIDATES]:
        action_desc = act.get("thought", "")
        if act.get("action_type") == "tool" and act.get("tool_name"):
            action_desc = f"[Tool: {act['tool_name']}] {act.get('thought', '')}"
        elif act.get("reasoning"):
            action_desc = f"[Reasoning] {act.get('thought', '')}"

        child = TreeNode(state_text=node.state_text, parent=node)
        child.action = action_desc
        child.action_type = act.get("action_type", "reasoning")
        child.tool_name = act.get("tool_name")
        child.tool_input = act.get("tool_input")
        child.reasoning = act.get("reasoning")
        node.children.append(child)
        candidates.append(child)

    print(f"[expand] 生成 {len(candidates)} 个候选动作:")
    for i, c in enumerate(candidates):
        print(f"  [{i}] {c.action[:80]}")

    return {"candidates": candidates}


def node_evaluate(state: LATSState) -> dict:
    """3. Evaluate — LLM Critic 对候选动作打分，选出最优。"""
    candidates: List[TreeNode] = state.get("candidates", [])
    if not candidates:
        return {"current_node": state["current_node"]}

    question = state["question"]
    node: TreeNode = state["current_node"]

    candidates_text = "\n".join(
        f"[{i}] {c.action}" for i, c in enumerate(candidates)
    )

    prompt = CRITIC_SYSTEM.format(
        question=question,
        state_text=node.state_text,
        candidates_text=candidates_text,
    )

    llm = _get_llm(temperature=0.1)
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content if isinstance(response.content, str) else json.dumps(response.content)

    try:
        scores = _parse_json_array(raw)
        for s in scores:
            idx = int(s.get("index", 0))
            if 0 <= idx < len(candidates):
                candidates[idx].heuristic_value = float(s.get("score", 0.5))
        print("[evaluate] 评分结果:")
        for s in scores:
            idx = int(s.get("index", 0))
            if 0 <= idx < len(candidates):
                print(f"  [{idx}] heuristic={s.get('score', '?')}: {s.get('reason', '')[:60]}")
    except Exception as e:
        print(f"[evaluate] 解析评分失败: {e}，使用默认评分")
        for c in candidates:
            c.heuristic_value = 0.5

    best = max(candidates, key=lambda c: c.heuristic_value)
    print(f"[evaluate] 选中最优候选: heuristic={best.heuristic_value:.2f}, action={best.action[:60]}")
    return {"current_node": best}


def node_simulate(state: LATSState) -> dict:
    """4. Simulate — 在真实环境中执行动作，获取观察结果。"""
    node: TreeNode = state["current_node"]
    question = state["question"]

    if node.action_type == "tool" and node.tool_name:
        te = _get_tool_executor()
        tool_input = node.tool_input or ""
        print(f"[simulate] 执行工具: {node.tool_name}({tool_input!r})")
        observation, success = te.execute(node.tool_name, tool_input)
        print(f"[simulate] 工具结果 (success={success}): {observation[:300]}")
    elif node.reasoning:
        llm = _get_llm(temperature=0.3)
        reasoning_prompt = (
            f"基于以下状态，进行推理分析：\n{node.state_text}\n\n"
            f"原始问题：{question}\n\n"
            f"推理方向：{node.reasoning}\n\n"
            f"请给出你的分析结果。"
        )
        resp = llm.invoke([HumanMessage(content=reasoning_prompt)])
        observation = resp.content if isinstance(resp.content, str) else str(resp.content)
        success = True
        print(f"[simulate] 推理结果: {observation[:300]}")
    else:
        observation = "No action executed."
        success = False

    node.state_text += f"\nAction: {node.action}\nObservation: {observation}"

    is_answer = _check_terminal(observation, question, node)
    if is_answer:
        node.is_terminal = True
        node.is_success = True
        node.reward = 1.0
        print("[simulate] 检测到成功终止!")
    elif not success:
        node.reward = -0.5
        print("[simulate] 工具执行失败，扣分")
    else:
        node.reward = 0.2
        print("[simulate] 执行成功，给予基础正奖励")

    best_node = state.get("best_node")
    if best_node is None or node.reward > best_node.reward:
        best_node = node

    return {"current_node": node, "best_node": best_node}


def _check_terminal(observation: str, question: str, node: TreeNode) -> bool:
    """通过 LLM 判断当前观察是否足以回答原始问题。"""
    if node.depth < 1:
        return False

    trajectory = _get_trajectory(node)
    llm = _get_llm(temperature=0.0)
    prompt = (
        f"原始问题：{question}\n\n"
        f"到目前为止的完整轨迹：\n{trajectory}\n\n"
        f"最新观察：{observation[:1000]}\n\n"
        f"基于以上信息，问题是否已经可以被完整回答？"
        f"只回答 YES 或 NO，不要解释。"
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    answer = resp.content.strip().upper() if isinstance(resp.content, str) else ""
    return "YES" in answer


def node_backpropagate(state: LATSState) -> dict:
    """5. Backpropagate — 将结果沿路径回传，更新 MCTS 统计信息。

    标准 MCTS：对路径上每个节点 visits += 1, value += reward。
    UCT 公式中 value/visits 自动计算均值，无需手动做增量平均。
    """
    node: TreeNode = state["current_node"]
    reward = node.reward

    print(f"[backprop] 回溯更新: reward={reward:.2f}")
    cur = node
    while cur is not None:
        cur.visits += 1
        cur.value += reward
        cur = cur.parent

    step = state.get("current_step", 0) + 1
    print(f"[backprop] 完成第 {step} 轮迭代")
    return {"current_step": step}


def node_reflect(state: LATSState) -> dict:
    """6. Reflect — 失败时生成反思，存入全局记忆池。"""
    node: TreeNode = state["current_node"]
    reflections = list(state.get("reflections", []))

    if node.is_terminal and not node.is_success:
        trajectory = _get_trajectory(node)
        question = state["question"]

        prompt = REFLECT_SYSTEM.format(question=question, trajectory=trajectory)
        llm = _get_llm(temperature=0.3)
        resp = llm.invoke([HumanMessage(content=prompt)])
        reflection = resp.content.strip() if isinstance(resp.content, str) else str(resp.content)
        reflections.append(reflection)
        print(f"[reflect] 新反思: {reflection[:100]}")
    elif node.is_success:
        print("[reflect] 成功路径，无需反思")
    else:
        print("[reflect] 非终止节点，跳过反思")

    return {"reflections": reflections}


# ---------------------------------------------------------------------------
# 路由逻辑
# ---------------------------------------------------------------------------


def should_continue(state: LATSState) -> str:
    """条件判断：是否继续搜索。"""
    current_step = state.get("current_step", 0)
    max_budget = state.get("max_budget", DEFAULT_BUDGET)
    node: TreeNode = state.get("current_node")

    if node and node.is_success:
        print(f"\n[router] 找到成功解! 在第 {current_step} 轮结束")
        return "generate_answer"
    if current_step >= max_budget:
        print(f"\n[router] 达到搜索预算上限 ({max_budget} 轮)，生成最佳答案")
        return "generate_answer"
    print(f"\n[router] 继续搜索 (step {current_step}/{max_budget})")
    return "select"


def node_generate_answer(state: LATSState) -> dict:
    """搜索结束后，根据最佳轨迹生成最终答案。"""
    question = state["question"]
    best = state.get("best_node") or state.get("current_node")
    trajectory = _get_trajectory(best) if best else "(no trajectory)"

    prompt = FINAL_ANSWER_SYSTEM.format(question=question, best_trajectory=trajectory)
    llm = _get_llm(temperature=0.1)
    resp = llm.invoke([HumanMessage(content=prompt)])
    answer = resp.content.strip() if isinstance(resp.content, str) else str(resp.content)
    print(f"\n[answer] 最终答案已生成")
    return {"answer": answer}


# ---------------------------------------------------------------------------
# 图构建
# ---------------------------------------------------------------------------


def build_graph(workspace_root: Optional[str] = None):
    global _tool_executor
    _tool_executor = ToolExecutor(workspace_root=workspace_root or os.getcwd())

    workflow = StateGraph(LATSState)

    workflow.add_node("select", node_select)
    workflow.add_node("expand", node_expand)
    workflow.add_node("evaluate", node_evaluate)
    workflow.add_node("simulate", node_simulate)
    workflow.add_node("backpropagate", node_backpropagate)
    workflow.add_node("reflect", node_reflect)
    workflow.add_node("generate_answer", node_generate_answer)

    workflow.set_entry_point("select")
    workflow.add_edge("select", "expand")
    workflow.add_edge("expand", "evaluate")
    workflow.add_edge("evaluate", "simulate")
    workflow.add_edge("simulate", "backpropagate")
    workflow.add_edge("backpropagate", "reflect")

    workflow.add_conditional_edges(
        "reflect",
        should_continue,
        {
            "select": "select",
            "generate_answer": "generate_answer",
        },
    )
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


def create_initial_state(question: str, max_budget: int = DEFAULT_BUDGET) -> LATSState:
    """创建初始状态，供 CLI 调用。"""
    root = TreeNode(state_text=f"Question: {question}")
    root.visits = 1
    return {
        "question": question,
        "root": root,
        "current_node": root,
        "candidates": [],
        "reflections": [],
        "max_budget": max_budget,
        "current_step": 0,
        "best_node": None,
        "answer": "",
    }
