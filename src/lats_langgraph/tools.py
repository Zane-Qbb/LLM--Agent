"""LATS 工具集 —— 复用 rewoo_langgraph 的基础工具，并封装为统一的执行接口。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rewoo_langgraph.tools import build_tools


@dataclass
class ToolExecutor:
    """统一的工具执行器，对外暴露 execute(tool_name, tool_input) -> (output, success)。"""

    workspace_root: str
    _tool_map: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self):
        tools = build_tools(self.workspace_root)
        self._tool_map = {t.name: t for t in tools}

    @property
    def tool_names(self) -> List[str]:
        return list(self._tool_map.keys())

    @property
    def tool_descriptions(self) -> str:
        lines = []
        for t in self._tool_map.values():
            lines.append(f"- {t.name}: {t.description}")
        return "\n".join(lines)

    def execute(self, tool_name: str, tool_input: str) -> Tuple[str, bool]:
        """执行工具，返回 (output_text, success_flag)。"""
        tool = self._tool_map.get(tool_name)
        if not tool:
            return f"UNKNOWN_TOOL: {tool_name}. Available: {', '.join(self.tool_names)}", False
        try:
            result = tool.invoke(tool_input)
            return str(result), True
        except Exception as e:
            return f"TOOL_ERROR({tool_name}): {e}", False
