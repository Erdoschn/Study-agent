"""Optional MCP adapter for Study Agent.

Install requirements-mcp.txt, start the Study Agent HTTP API separately, then
run this module over stdio from an MCP-compatible client.
"""
from __future__ import annotations

import json
import sys
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - depends on optional extra
    raise SystemExit(
        "MCP adapter dependency missing. Install it with: "
        "python -m pip install -r requirements-mcp.txt"
    ) from exc

from integrations.study_agent_client import StudyAgentClient

mcp = FastMCP("study-agent")


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


@mcp.tool()
def study_agent_ask(username: str, question: str, context: str = "") -> str:
    """Ask Study Agent to solve or explain a learning task for a named profile.

    The username selects a data partition; it is not an authentication credential.
    Use context only for relevant course material or conversation context.
    This tool does not provide arbitrary shell, filesystem, or browser control.
    """
    return _json_text(
        StudyAgentClient().ask(question, username=username, context=context or None)
    )


@mcp.tool()
def study_agent_health() -> str:
    """Check whether the configured Study Agent API is reachable."""
    return _json_text(StudyAgentClient().health())


if __name__ == "__main__":
    # MCP stdio transport reserves stdout for protocol messages.
    print("Starting Study Agent MCP adapter (stdio).", file=sys.stderr)
    mcp.run(transport="stdio")
