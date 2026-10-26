"""Deterministic, read-only MCP fixtures for development and threat-model research."""

import sys
import time

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "Blueprint demo", host="0.0.0.0", port=8001, stateless_http=True, json_response=True
)

POLICY = "Blueprint demo policy: retain demo documents for 30 days. This is synthetic test data."


@mcp.tool()
def get_retention_policy() -> str:
    """Return the synthetic document retention policy."""
    return POLICY


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers without modifying state."""
    return a + b


@mcp.resource("blueprint://policy/retention")
def retention_policy() -> str:
    """Synthetic document retention policy."""
    return POLICY


@mcp.tool()
def slow_read(seconds: float = 2) -> str:
    """Simulate a bounded slow read for timeout testing; not enabled by default."""
    time.sleep(min(max(seconds, 0), 5))
    return POLICY


if __name__ == "__main__":
    mcp.run(transport="stdio" if "--stdio" in sys.argv else "streamable-http")
