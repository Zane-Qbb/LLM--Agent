import asyncio

from dotenv import load_dotenv

from .agent import build_graph, create_initial_state


async def _amain():
    load_dotenv()
    graph = build_graph()

    print("LATS (Language Agent Tree Search) CLI. 输入问题，回车发送；输入 exit 退出。")
    print("示例问题（适合需要多路径试错探索的任务）：")
    print("  - 这个项目里有没有潜在的安全漏洞？检查代码中的敏感信息处理、输入验证和文件访问控制")
    print("  - 分析这个项目的架构设计，找出最大的技术债务，并给出重构优先级建议")
    print()

    while True:
        question = input("\n> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break

        print(f"\nQuestion: {question}")
        print("=" * 60)

        initial_state = create_initial_state(question, max_budget=8)

        print("--- LATS Agent (MCTS 树搜索) ---\n")
        final_state = None
        for step in graph.stream(initial_state):
            for node_name, node_state in step.items():
                print(f"--- Node: {node_name} ---")
                if "answer" in node_state and node_state["answer"]:
                    print(f"\n{'='*60}")
                    print("FINAL ANSWER:")
                    print(f"{'='*60}")
                    print(node_state["answer"])
                    print(f"{'='*60}")
                final_state = node_state

        if final_state and "reflections" in final_state and final_state["reflections"]:
            print(f"\n--- 搜索过程中产生的反思 ({len(final_state['reflections'])} 条) ---")
            for i, r in enumerate(final_state["reflections"], 1):
                print(f"  [{i}] {r}")

        print()


def main():
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
