from __future__ import annotations

import asyncio
import json
import os

from dotenv import load_dotenv

from .agent import build_graph


async def _amain():
    load_dotenv()
    app = build_graph(workspace_root=os.getcwd())
    print("ReWOO LangGraph CLI. 输入问题，回车发送；输入 exit 退出。")
    while True:
        q = input("\n> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            break
        out = await app.ainvoke({"question": q})

        plan = out.get("plan")
        if plan:
            print("\n--- plan (from state) ---")
            print(json.dumps(plan, ensure_ascii=False, indent=2))

        observations = out.get("observations")
        if observations:
            print("\n--- observations (tool outputs) ---")
            print(json.dumps(observations, ensure_ascii=False, indent=2))

        print("\n--- answer ---")
        print(out.get("answer", ""))


def main():
    asyncio.run(_amain())


if __name__ == "__main__":
    main()

