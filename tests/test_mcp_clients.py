"""Contract tests for MCP clients.

These spawn the real upstream MCP servers. Skipped when the sibling-repo
path env vars are not set, so the suite stays green on machines without
the upstreams checked out.
"""

from __future__ import annotations

import os

import pytest

from src.config import get_settings
from src.mcp_clients.long_term import EXPECTED_TOOLS as LT_TOOLS
from src.mcp_clients.long_term import LongTermClient
from src.mcp_clients.short_term import EXPECTED_TOOLS as ST_TOOLS
from src.mcp_clients.short_term import ShortTermClient


def _has_path(attr: str) -> bool:
    s = get_settings()
    path = getattr(s, attr, "") or os.environ.get(attr.upper(), "")
    return bool(path) and os.path.isdir(path)


@pytest.mark.skipif(not _has_path("short_term_trader_path"),
                     reason="SHORT_TERM_TRADER_PATH not set or missing")
def test_short_term_mcp_exposes_expected_tools():
    c = ShortTermClient()
    c.start()
    try:
        names = set(c.list_tool_names())
    finally:
        c.stop()
    missing = ST_TOOLS - names
    assert not missing, f"upstream short-term-trader missing tools: {missing}"


@pytest.mark.skipif(not _has_path("stock_recommender_path"),
                     reason="STOCK_RECOMMENDER_PATH not set or missing")
def test_long_term_mcp_exposes_expected_tools():
    c = LongTermClient()
    c.start()
    try:
        names = set(c.list_tool_names())
    finally:
        c.stop()
    missing = LT_TOOLS - names
    assert not missing, f"upstream stock-recommender missing tools: {missing}"
