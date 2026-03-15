from __future__ import annotations

import asyncio
import json
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from .agent import _ainvoke_text, _extract_json, _get_llm, _normalize_steps, build_executor_graph
from .prompts import PLANNER_SYSTEM, Plan


async def _amain():
    load_dotenv()
    workspace_root = os.getcwd()
    executor_app = build_executor_graph(workspace_root=workspace_root)

    planner_llm = _get_llm("PLANNER_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))

    print("Plan&Execute LangGraph CLI（含人工确认 Plan）.")
    print("提示：设置 STREAM_LLM=1 可开启 planner/judge/replanner 流式输出。")
    while True:
        q = input("\n> 目标/问题：").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            break

        # 1) 先只做规划
        msg = [SystemMessage(content=PLANNER_SYSTEM), HumanMessage(content=q)]
        text = await _ainvoke_text(planner_llm, msg, label="planner")
        raw = _extract_json(text)
        plan_obj = Plan.model_validate(raw)
        steps = _normalize_steps(list(plan_obj.steps))

        print("\n=== Planner 生成的初始 Plan ===")
        for i, s in enumerate(steps, 1):
            print(f"{i}. {s}")

        # 2) 人工确认/修改
        choice = input("\n确认执行这个计划吗？(y=直接执行, e=手动编辑, r=重新规划, q=放弃) ").strip().lower()
        if choice == "q":
            print("已放弃本次任务。")
            continue
        if choice == "r":
            print("请重新输入问题触发新一轮规划。")
            continue
        if choice == "e":
            print("逐行输入你修改后的步骤，空行结束：")
            new_steps = []
            while True:
                line = input()
                if not line.strip():
                    break
                new_steps.append(line.strip())
            if new_steps:
                steps = new_steps

        if not steps:
            print("计划为空，终止执行。")
            continue

        # 3) 执行确认后的计划
        print("\n=== 开始按照确认后的 Plan 执行 ===")
        init_state = {
            "objective": q,
            "original_plan": steps,
            "plan": steps,
            "past_steps": [],
        }

        last_past_len = 0
        out = None
        async for state in executor_app.astream(init_state, stream_mode="values"):
            out = state
            past_steps = state.get("past_steps") or []
            if len(past_steps) > last_past_len:
                print("\n--- progress (new step finished) ---")
                print(json.dumps(past_steps[-1], ensure_ascii=False, indent=2))
                last_past_len = len(past_steps)

        if out is None:
            print("执行过程中发生未知错误（未获得最终状态）。")
            continue

        remaining_plan = out.get("plan")
        if remaining_plan:
            print("\n--- remaining plan ---")
            print(json.dumps(remaining_plan, ensure_ascii=False, indent=2))

        past_steps = out.get("past_steps")
        if past_steps:
            print("\n--- past steps ---")
            print(json.dumps(past_steps, ensure_ascii=False, indent=2))

        answer = out.get("answer")
        print("\n--- answer ---")
        print(answer or "(no final answer yet)")


def main():
    asyncio.run(_amain())


if __name__ == "__main__":
    main()

