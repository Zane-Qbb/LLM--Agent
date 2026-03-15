# LangGraph ReWOO Agent（Plan→Tools→Solve）

这是一个用 **LangGraph** 实现的 **ReWOO（Reasoning Without Observation）模式** agent 示例工程：
先产出可执行计划（Plan），再按步骤调用工具（Tools/Worker），最后汇总得到答案（Solve）。

另外，本仓库还包含一个 **Plan & Execute** 范式的 agent（带 **Replanner 动态重规划**）：

- **Planner**：把 Goal 拆成步骤列表
- **Executor（ReAct Agent）**：只执行当前一步，挂载工具，遇到异常会自我修复/重试
- **Replanner**：基于已完成步骤与结果，决定继续 / 改计划 / 直接终止并返回答案

本仓库还包含一个 **LLM Compiler** 范式的 agent（带 **动态 DAG 重规划 + 局部容错**）：

- **Function Calling Planner**：将自然语言查询编译成 DAG（有向无环图）的纯工具调用结构
- **Task Fetching Unit**：基于拓扑排序的动态执行器，支持并行批次执行与局部容错
- **Executor**：统一执行工具调用（含 `llm_reasoning`、`join` 等 LLM 封装工具）
- **Controller**：决策是否继续迭代或返回最终答案
- **Local Replan**：节点失败时，冻结下游子树，异步触发局部重规划并热合并替代子 DAG

本仓库还包含一个 **Self-Discover** 范式的 agent（基于 LangGraph）：

- **Select**：根据具体任务，从多个推理模块（Reasoning Modules）中挑选出相关的模块
- **Adapt**：将挑选出的通用推理模块，改编为针对当前任务的具体指导原则
- **Implement**：将改编后的模块，转化为一个可执行的推理结构（Reasoning Structure）
- **Execute**：使用该推理结构，一步步推导并得出最终答案

## 1) 安装与运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

安装本项目（让 `python -m rewoo_langgraph.cli` 能直接找到包）：

```bash
pip install -e .
```

准备模型（任选其一）：

- **OpenAI 兼容**（推荐）：设置环境变量

```bash
export OPENAI_API_KEY="your-api-key-here"
# 可选：兼容 OneAPI / vLLM / 其他网关
export OPENAI_BASE_URL="https://xxx"
export OPENAI_MODEL="xxx"
```

运行交互式 CLI：

```bash
python -m rewoo_langgraph.cli
```

运行 Plan & Execute 交互式 CLI：

```bash
python -m plan_execute_langgraph.cli
```

运行 LLM Compiler 交互式 CLI：

```bash
python -m llm_compiler_langgraph.cli
```

运行 Self-Discover 交互式 CLI：

```bash
python -m self_discover_langgraph.cli
```

## 2) 已挂载工具（你怎么问会触发）

本项目内置了这些工具（planner 会决定要不要用）：

- **`calculator`**：数学计算、单位换算里需要算式的部分  
  - 触发示例：  
    - “帮我算 \( (123.5*18-36)/7 \) 并给出结果”  
    - “把 3.2GB 换算成 MB（按 1024 进制）并计算”

- **`now`**：取当前时间（含时区）  
  - 触发示例：  
    - “现在北京时间几点？再给我一个 ISO8601 时间戳”

- **`list_dir`**：列出某个目录下的文件（默认 workspace 根目录）  
  - 触发示例：  
    - “列一下当前项目根目录有哪些文件/子目录”

- **`read_text_file`**：读取本地文本文件（只允许读 workspace 内）  
  - 触发示例：  
    - “把 `README.md` 的前 80 行总结一下”  
    - “读取 `requirements.txt`，告诉我用了哪些包”

- **`web_get`**（可选网络）：抓取一个网页的纯文本（适合读取文档片段）  
  - 触发示例：  
    - “抓取 `https://python.langchain.com/docs/` 首页标题并总结 3 点”

## 3) 最稳的“提问模板”（建议你直接照抄）

为了让 planner 更稳定地触发工具，你可以用下面这种格式：

### 模板 A：强制工具

> 目标：xxx  
> 允许工具：calculator, read_text_file  
> 输入：...  
> 输出格式：...

### 模板 B：分步任务（最适合 ReWOO）

> 我需要你先给出计划（每步写清要用的工具和输入），再执行工具，最后汇总答案：  
> 任务：xxx

### 模板 C：本地文件分析

> 请读取文件 `path/to/file`（只读文本），提取关键字段并输出为 JSON：...

## 3.5) LLM Compiler 使用说明

### 什么是 LLM Compiler？

LLM Compiler 是一种将复杂查询**编译成 DAG（有向无环图）**的 agent 范式（参考 [ICML 2024 论文](https://arxiv.org/abs/2312.04511)）：

1. **Function Calling Planner** 将问题编译成纯工具调用 DAG：
   - 所有节点都是 tool 调用（无 thought 节点）
   - LLM 推理封装为 `llm_reasoning` 工具
   - 最终聚合封装为 `join` 工具
   - 节点间通过 `$nX` 占位符传递数据

2. **Task Fetching Unit** 动态执行 DAG：
   - 拓扑排序 + 并行批次执行
   - **局部容错**：节点失败时冻结下游子树，触发局部重规划
   - **热合并**：替代子 DAG 实时注入正在执行的主 DAG
   - 成功的并行分支不受失败影响，继续执行

3. **Controller** 进行动态重规划：
   - 评估当前结果是否足够
   - 决定继续迭代或返回最终答案

### 与 ReWOO / Plan & Execute 的区别

| 范式 | 规划方式 | 执行方式 | 适用场景 |
|------|----------|----------|----------|
| ReWOO | 线性计划 | 顺序执行 | 简单任务，步骤明确 |
| Plan & Execute | 步骤列表 | 单步执行 + 动态调整 | 需要灵活应对的任务 |
| LLM Compiler | DAG 图 | 拓扑排序 + 并行 | 多分支、可并行的复杂任务 |

### LLM Compiler 配置选项

```bash
# 可选：为不同组件指定不同模型
export LLM_COMPILER_PLANNER_MODEL="qwen3.5-plus"   # 规划器模型
export LLM_COMPILER_THOUGHT_MODEL="qwen3.5-plus"  # 推理模型
export LLM_COMPILER_CONTROLLER_MODEL="qwen3.5-plus" # 控制器模型

# 可选：最大重试次数（默认 3 次）
export OPENAI_MAX_RETRIES="3"
```

### 鲁棒性特性

LLM Compiler 内置了多项生产级鲁棒性保障：

| 特性 | 说明 | 实现机制 |
|------|------|----------|
| **环路检测** | 防止 Planner 生成循环依赖的 DAG | Kahn 算法，O(V+E) 时间复杂度 |
| **自我修正** | JSON 格式错误或字段验证失败时自动重试 | 将错误信息反馈给 Planner 要求重新生成 |
| **命名空间隔离** | 防止多轮迭代中节点 ID 覆盖（如 $1 覆盖 $1） | 每代添加 `g{iteration}_` 前缀，args 占位符同步更新 |
| **局部容错重规划** | 节点失败时只重规划失败子树，不影响成功分支 | 冻结下游 + 异步局部重规划 + 热合并替代子 DAG |
| **混合错误检测** | 自动识别工具执行失败（含模糊场景） | 前缀快判 + LLM 模糊判断双层策略 |
| **Fallback 机制** | 达到最大重试次数仍失败时的降级策略 | 返回简化 DAG 仅执行总结推理 |

#### 环路检测机制

当 Planner 生成 $A$ 依赖 $B$，$B$ 依赖 $A$ 的循环依赖时：

1. Task Fetching Unit 在接收到 DAG 后第一时间进行环路检测
2. 一旦检测到环，抛出 `CycleDetectedError`
3. 触发 Fallback 机制，将产生环的局部图作为 Error Message 返回给 Planner
4. Planner 根据错误信息自我修正，重新生成无环的调度计划

```
[compiler.planner] 验证失败：Cycle detected involving nodes: ['n2', 'n3', 'n1']
[compiler.planner] 第 1 次尝试失败：Cycle detected involving nodes: ['n2', 'n3', 'n1']
【上次规划错误】Cycle detected involving nodes: ['n2', 'n3', 'n1']
请修正上述问题，重新生成有效的 DAG 计划。
```

#### 命名空间隔离（Generation Isolation）

场景：Replan 后，Planner 重新从 `$1` 开始编号，可能覆盖第一轮迭代的数据。

修复方案：引擎层对每代节点 ID 添加代际前缀：

```
Iteration 1: n1, n2, n3  →  g1_n1, g1_n2, g1_n3
Iteration 2: n1, n2      →  g2_n1, g2_n2  （不会覆盖 g1_n1）
```

这样即使 Planner 生成的 ID 重复，实际内存中也不会发生冲突。

#### 局部容错重规划（Local Fault-Tolerant Replan）

当 DAG 中某个节点执行失败时（如 `web_get` 返回 502），传统做法是等整个 DAG 跑完再由 Controller 决定全局重规划。而局部容错机制能做到：

**场景示例**：

```
A: 搜索北京天气  ──→  C: 北京穿搭建议  ──→  join
B: 搜索上海天气  ──→  D: 上海穿搭建议  ──→  join
```

假设 A 执行失败：

1. **B/D 继续执行**：成功的并行分支不受影响
2. **冻结 C**：A 的下游子树被冻结，不会拿到错误数据
3. **异步局部重规划**：Planner 收到失败信息，生成替代子 DAG（如用 `llm_reasoning` 代替 `web_get`）
4. **热合并**：替代子 DAG（A'/C'）被注入正在执行的主 DAG，join 节点的依赖自动重连

```
[compiler.executor] *** 节点 g1_n1 执行失败，触发局部重规划 ***
  冻结下游节点：{'g1_n3'}
[compiler.replan] 开始局部重规划，失败节点：g1_n1
[compiler.replan] 局部重规划完成，新节点数：2
[compiler.replan.merge] 重连 g1_n5: deps ['g1_n3', 'g1_n4'] -> ['g1_n4', 'g1_r1_r1', 'g1_r1_r2']
```

**保护机制**：
- 每次 DAG 执行最多允许 2 次局部重规划（`max_local_replans`）
- 局部重规划失败时降级为旧行为（错误数据写入 memory，继续执行）
- `join` 节点的失败不触发局部重规划

### 日志输出说明

LLM Compiler 输出详细日志，包括：

- `[compiler.planner]` - DAG 规划阶段
  - 规划耗时
  - DAG 结构和依赖关系图
  - 每个节点的描述和参数

- `[compiler.executor]` - DAG 执行阶段
  - 每批并行执行的节点
  - 每个节点的耗时和输出
  - Memory 更新和依赖满足情况
  - 失败检测和冻结下游子树

- `[compiler.replan]` - 局部重规划阶段
  - 失败节点信息和冻结的下游节点
  - 替代子 DAG 结构
  - 热合并过程（节点注入、依赖重连）

- `[compiler.controller]` - 决策阶段
  - 决策结果（continue/final）
  - 决策理由

### LLM Compiler 示例问题

```bash
# 24 点游戏
python -m llm_compiler_langgraph.cli
> 用 3,3,8,8 凑出 24 点，给出所有可能的解法

# 多分支信息检索
> 比较 Python 和 Rust 的内存管理机制，需要查阅官方文档
```

## 3.6) Self-Discover 使用说明

### 什么是 Self-Discover？

Self-Discover 是一种让 LLM 在解决复杂推理任务时，自动发现并组装适合当前任务的推理结构（Reasoning Structure）的框架（参考 [Google DeepMind 论文](https://arxiv.org/abs/2402.03620)）。

它包含两个阶段，四个步骤：

**Stage 1: Discover**
1. **Select**：给定一个任务描述，从 39 个预定义的通用推理模块（如“把问题分解”、“逆向思考”、“批判性思维”）中，挑选出最相关的几个。
2. **Adapt**：将挑选出的通用模块，结合当前任务的具体情境进行改编，使其更具针对性。
3. **Implement**：将改编后的模块，组织并实现为一个结构化的、可执行的推理计划（Reasoning Structure）。

**Stage 2: Execute**
4. **Execute**：LLM 遵循生成的推理结构，一步步推导并得出最终答案。

### 与其他框架的区别

- **ReWOO / Plan & Execute**：侧重于**工具调用**的规划与执行。
- **Self-Discover**：侧重于**纯逻辑推理**任务，它不调用外部工具，而是通过元推理（Meta-Reasoning）找到最适合当前问题的思考方式。

### 示例问题

Self-Discover 非常适合解决复杂的逻辑谜题、数学应用题或需要多步推理的场景。

```bash
python -m self_discover_langgraph.cli
> 一个农场里有鸡、鸭、鹅、猪、牛、羊。已知：
1. 鸡和鸭的数量之和是鹅的3倍。
2. 猪和牛的数量之和是羊的2倍。
3. 鸡、鸭、鹅的总腿数是猪、牛、羊总腿数的1.5倍。
4. 鸭的数量比鸡多5只，牛的数量比猪少2头。
5. 农场里所有动物的总头数是 44 个。
请问：农场里每种动物各有多少只（头）？
```

## 4) 代码结构

- `src/rewoo_langgraph/agent.py`：LangGraph 主流程（Plan→Act→Solve）
- `src/rewoo_langgraph/tools.py`：工具定义与安全限制
- `src/rewoo_langgraph/prompts.py`：planner/solver 提示词与输出 schema
- `src/rewoo_langgraph/cli.py`：命令行入口

- `src/plan_execute_langgraph/agent.py`：LangGraph 主流程（Plan→Execute(循环)→Solve）
- `src/plan_execute_langgraph/prompts.py`：planner/solver 提示词与输出 schema
- `src/plan_execute_langgraph/cli.py`：命令行入口
- `src/plan_execute_langgraph/tools.py`：复用 `rewoo_langgraph` 的工具集

- `src/llm_compiler_langgraph/agent.py`：LLM Compiler 主流程（动态 DAG 执行 + 局部容错重规划）
- `src/llm_compiler_langgraph/prompts.py`：planner/controller/replan 提示词与输出 schema
- `src/llm_compiler_langgraph/cli.py`：命令行入口

- `src/self_discover_langgraph/agent.py`：Self-Discover 主流程（Select→Adapt→Implement→Execute）
- `src/self_discover_langgraph/prompts.py`：预定义的推理模块
- `src/self_discover_langgraph/cli.py`：命令行入口

## 5) FAQ

### 为什么 LLM Compiler 不使用 LangGraph 实现？

LLM Compiler 的核心设计是 **动态 DAG 生成 + 拓扑排序执行**，这与 LangGraph 的设计理念有所不同：

| 特性 | LangGraph | LLM Compiler 自定义实现 |
|------|-----------|------------------------|
| 图结构 | 预定义的状态机/工作流 | 动态生成的 DAG |
| 执行方式 | 基于边的条件转移 | 拓扑排序 + 并行批次 |
| 灵活性 | 适合固定流程 | 适合每次迭代图结构都变化的场景 |
| 并行性 | 需要显式定义 | 自动从无依赖节点发现 |

简单来说：
- **LangGraph** 适合：状态明确、流程固定的场景（如 ReWOO 的 Plan→Act→Solve）
- **自定义实现** 适合：每次迭代生成的图结构都不同、需要动态拓扑排序的场景

当然，理论上也可以用 LangGraph 的 `StateGraph` + 动态条件边来实现，但会增加复杂度，而自定义的拓扑排序实现更直观、高效。

### LLM Compiler 如何处理 Planner 生成的错误 DAG？

LLM Compiler 内置了多层防御机制：

1. **JSON 解析层**：如果模型输出的不是合法 JSON，捕获解析错误并触发重试
2. **Pydantic 验证层**：如果 JSON 结构不符合 `DagPlan` schema，验证失败并触发重试
3. **语义验证层**：检查环路、自依赖、无效依赖等，使用 Kahn 算法进行 O(V+E) 检测
4. **Fallback 层**：达到最大重试次数（默认 3 次）后，返回简化的 summary DAG 作为降级

每次重试时，错误信息会附加到 Prompt 中，让 Planner 进行自我修正。

