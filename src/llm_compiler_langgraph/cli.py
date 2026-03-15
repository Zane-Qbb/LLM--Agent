from __future__ import annotations

import asyncio
import json
import os

from dotenv import load_dotenv

from .agent import run_compiler


async def _amain():
    load_dotenv()
    workspace_root = os.getcwd()
    print("LLM Compiler LangGraph CLI（含动态重规划 / Replan 外层循环）。")
    print("输入问题（例如：用 3,3,8,8 凑 24 点），回车发送；输入 exit 退出。")

    while True:
        q = input("\n> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            break

        result = await run_compiler(q, workspace_root=workspace_root)

        memory = result.get("memory") or {}
        iterations = result.get("iterations")
        answer = result.get("answer") or ""

        print(f"\n=== outer iterations ===\n{iterations}")
        if memory:
            print("\n=== memory (all node outputs) ===")
            print(json.dumps(memory, ensure_ascii=False, indent=2))

        print("\n=== final answer ===")
        print(answer)


def main():
    asyncio.run(_amain())


if __name__ == "__main__":
    main()

