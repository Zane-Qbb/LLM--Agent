import asyncio
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from .agent import build_graph, _get_llm


async def _amain():
    load_dotenv()
    graph = build_graph()
    llm = _get_llm()
    
    print("Self-Discover LangGraph CLI. 输入问题，回车发送；输入 exit 退出。")
    print("示例问题：一个农场里有鸡、鸭、鹅、猪、牛、羊。已知：\n1. 鸡和鸭的数量之和是鹅的3倍。\n2. 猪和牛的数量之和是羊的2倍。\n3. 鸡、鸭、鹅的总腿数是猪、牛、羊总腿数的1.5倍。\n4. 鸭的数量比鸡多5只，牛的数量比猪少2头。\n5. 农场里所有动物的总头数是 44 个。\n请问：农场里每种动物各有多少只（头）？\n")
    
    while True:
        task = input("\n> ").strip()
        if not task:
            continue
        if task.lower() in {"exit", "quit", "q"}:
            break
            
        print(f"\nTask: {task}\n")
        print("=" * 50)
        
        # 运行 Self-Discover 流程
        print("--- Self-Discover Agent ---")
        for step in graph.stream({"task_description": task}):
            for node_name, state in step.items():
                print(f"--- Node: {node_name} ---")
                if "selected_modules" in state:
                    print(f"Selected Modules:\n{state['selected_modules']}\n")
                elif "adapted_modules" in state:
                    print(f"Adapted Modules:\n{state['adapted_modules']}\n")
                elif "reasoning_structure" in state:
                    print(f"Reasoning Structure:\n{state['reasoning_structure']}\n")
                elif "solution" in state:
                    print(f"Solution:\n{state['solution']}\n")
                print("=" * 50)

        # 先获取 baseline (普通 LLM 直接回答)
        print("--- Baseline (Direct LLM Answer) ---")
        try:
            baseline_response = llm.invoke([HumanMessage(content=task)])
            print(f"{baseline_response.content.strip()}\n")
        except Exception as e:
            print(f"Error getting baseline response: {e}\n")
        print("=" * 50)

def main():
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
