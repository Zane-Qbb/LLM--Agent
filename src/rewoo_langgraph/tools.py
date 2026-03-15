from __future__ import annotations

import ast
import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from langchain_core.tools import tool


@dataclass(frozen=True)
class Workspace:
    root: Path

    def resolve_in_workspace(self, p: str) -> Path:
        raw = Path(p)
        target = (self.root / raw) if not raw.is_absolute() else raw
        resolved = target.resolve()
        root_resolved = self.root.resolve()
        if root_resolved not in resolved.parents and resolved != root_resolved:
            raise ValueError(f"Path escapes workspace: {p}")
        return resolved


def _safe_eval_arithmetic(expr: str) -> float:
    """
    Evaluate arithmetic safely: + - * / ** () and numbers.
    """

    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Mod,
        ast.FloorDiv,
        ast.Constant,
    )

    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError("Only arithmetic expressions are allowed.")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("Only int/float constants are allowed.")
    return float(eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, {}))


def build_tools(workspace_root: str) -> List[Any]:
    ws = Workspace(root=Path(workspace_root))

    @tool("calculator")
    def calculator(expression: str) -> str:
        """Evaluate a pure arithmetic expression and return the numeric result."""
        val = _safe_eval_arithmetic(expression.strip())
        if abs(val - round(val)) < 1e-12:
            return str(int(round(val)))
        return str(val)

    @tool("now")
    def now(tz: Optional[str] = None) -> str:
        """Get current time. Optionally provide tz like 'Asia/Shanghai'."""
        # Avoid extra dependencies; use system local time, or simple UTC if tz given.
        if tz:
            # Best-effort: if zoneinfo available.
            try:
                from zoneinfo import ZoneInfo  # py3.9+

                z = ZoneInfo(tz)
                t = dt.datetime.now(tz=z)
            except Exception as e:
                return f"tz not supported ({tz}): {e}"
        else:
            t = dt.datetime.now()
        return t.isoformat()

    @tool("list_dir")
    def list_dir(path: Optional[str] = None, max_entries: int = 200) -> str:
        """List a directory under the workspace. Default is workspace root."""
        target = ws.root if not path else ws.resolve_in_workspace(path)
        if not target.exists():
            return f"NOT_FOUND: {target}"
        if not target.is_dir():
            return f"NOT_A_DIR: {target}"
        entries = []
        for p in sorted(target.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            suffix = "/" if p.is_dir() else ""
            entries.append(p.name + suffix)
            if len(entries) >= max_entries:
                entries.append("... (truncated)")
                break
        return "\n".join(entries)

    @tool("read_text_file")
    def read_text_file(path: str, max_chars: int = 12000) -> str:
        """Read a UTF-8 text file under the workspace and return its content (truncated)."""
        p = ws.resolve_in_workspace(path)
        if not p.exists():
            return f"NOT_FOUND: {p}"
        if p.is_dir():
            return f"IS_A_DIR: {p}"
        data = p.read_text(encoding="utf-8", errors="replace")
        if len(data) > max_chars:
            return data[:max_chars] + "\n... (truncated)"
        return data

    @tool("web_get")
    def web_get(url: str, timeout_s: int = 20) -> str:
        """Fetch a web page via HTTP GET and return plain text (best-effort)."""
        raise RuntimeError("502 暂时无法提供服务（打桩模拟）")  # 打桩模拟工具调用失败，触发replan
        headers = {"User-Agent": "rewoo-langgraph/0.1 (+https://example.invalid)"}
        r = requests.get(url, headers=headers, timeout=timeout_s)
        r.raise_for_status()
        text = r.text
        text = " ".join(text.split())
        return text[:12000] + ("... (truncated)" if len(text) > 12000 else "")

    return [calculator, now, list_dir, read_text_file, web_get]

