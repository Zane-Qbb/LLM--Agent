from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Plan(BaseModel):
    steps: List[str] = Field(..., description="Remaining steps, in order. Each step is a concise instruction.")


class JudgeDecision(BaseModel):
    action: Literal["continue", "replan", "final"] = Field(
        ...,
        description="continue: execute next remaining step as-is; replan: invoke replanner to update remaining steps; final: answer now",
    )
    reason: Optional[str] = Field(default=None, description="Brief reason for the decision.")
    answer: Optional[str] = Field(default=None, description="Only when action='final': final answer to user.")


class ReplanDecision(BaseModel):
    action: Literal["plan", "final"] = Field(..., description="'plan' to continue with an updated plan, 'final' to answer user")
    steps: Optional[List[str]] = Field(default=None, description="Only when action='plan': remaining steps to do (do NOT include completed steps).")
    answer: Optional[str] = Field(default=None, description="Only when action='final': final answer to user.")


PLANNER_SYSTEM = """For the given objective, come up with a simple step by step plan.
This plan should involve individual tasks, that if executed correctly will yield the correct answer.
Do not add any superfluous steps.
The result of the final step should be the final answer.
Make sure that each step has all the information needed - do not skip steps.

Output ONLY valid JSON matching this schema:
{ "steps": ["1. ...", "2. ...", "..."] }
"""


EXECUTOR_SYSTEM = """You are an executor for a Plan & Execute agent.
You DO NOT need the full global plan. You only execute the CURRENT task.
You may use tools when helpful. Be robust: if a tool call fails, try reasonable fixes or alternatives.
Return a concise completion note for the task, including any key facts for the replanner.
"""


JUDGE_SYSTEM = """You are a judge/controller for a Plan & Execute agent.
You will be given:
- the overall objective
- the current task that was just executed
- the executor result for that task
- the remaining steps (as currently planned)

Decide whether we should:
- continue executing the next step without changing the plan,
- replan (update the remaining steps) because results indicate the plan should change,
- or finish early with a final answer.

Guidelines:
- Default to "continue" if the executor result looks successful and the remaining steps still make sense.
- Use "replan" if a tool failure, missing info, wrong direction, or new constraints require changing remaining steps.
- Use "final" only if the objective is already satisfied.

Output ONLY valid JSON matching this schema:
{ "action": "continue|replan|final", "reason": "...", "answer": "..." }
"""


REPLANNER_SYSTEM = """For the given objective, come up with a simple step by step plan.
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps.
The result of the final step should be the final answer. Make sure that each step has all the information needed – do not skip steps.

Your objective was this:
{objective}

Your original plan was this:
{plan}

You have currently done the following steps:
{past_steps}

Your remaining steps are currently:
{remaining_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that.
Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done.
Do not return previously done steps as part of the plan.

Output ONLY valid JSON matching one of the following schemas:
- To continue planning:
  {{ "action": "plan", "steps": ["...", "..."] }}
- To finish:
  {{ "action": "final", "answer": "..." }}
"""

