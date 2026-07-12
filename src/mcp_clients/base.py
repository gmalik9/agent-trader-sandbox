"""Sync wrapper around the official `mcp` async client.

The MCP SDK is asyncio-native; our broker/agent code is sync. This module
runs a single asyncio loop in a background thread per `MCPClient` instance
and exposes a sync `call(tool, **args)` facade.

The subprocess is launched as `python -m <module>` in the sibling repo's
`cwd`, with a curated env subset passed through.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from concurrent.futures import Future
from contextlib import suppress
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

log = logging.getLogger(__name__)


class MCPClientError(RuntimeError):
    pass


class MCPClient:
    """Long-lived sync facade over an MCP stdio subprocess."""

    def __init__(
        self,
        *,
        cwd: Path,
        module: str = "mcp_server.server",
        env_passthrough: tuple[str, ...] = (),
        extra_env: dict[str, str] | None = None,
        python: str | None = None,
        call_timeout: float = 30.0,
    ) -> None:
        self.cwd = Path(cwd).resolve()
        self.module = module
        self.env_passthrough = tuple(env_passthrough)
        self.extra_env = dict(extra_env or {})
        # Interpreter selection precedence:
        #   1. explicit `python=` arg (tests / callers)
        #   2. $MCP_PYTHON env override (Docker sets this to the container python,
        #      because a mounted host `.venv` is the wrong platform)
        #   3. the sibling repo's own `.venv/bin/python` (native local runs)
        #   4. `python3` on PATH
        if python is None:
            env_python = os.environ.get("MCP_PYTHON")
            if env_python:
                self.python = env_python
            else:
                sibling_venv = self.cwd / ".venv" / "bin" / "python"
                self.python = str(sibling_venv) if sibling_venv.exists() else "python3"
        else:
            self.python = python
        self.call_timeout = call_timeout

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._session: ClientSession | None = None
        self._stdio_ctx = None
        self._session_ctx = None
        self._failures = 0

    # ---------- lifecycle ----------

    def _build_env(self) -> dict[str, str]:
        env: dict[str, str] = {"PATH": os.environ.get("PATH", "")}
        for k in self.env_passthrough:
            v = os.environ.get(k)
            if v:
                env[k] = v
        env.update(self.extra_env)
        # PYTHONPATH ensures the sibling repo's own modules import.
        env["PYTHONPATH"] = str(self.cwd) + os.pathsep + env.get("PYTHONPATH", "")
        return env

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if not self.cwd.exists():
            raise MCPClientError(f"cwd does not exist: {self.cwd}")

        self._ready.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True,
                                         name=f"mcp:{self.cwd.name}")
        self._thread.start()
        if not self._ready.wait(timeout=20):
            raise MCPClientError(f"MCP server failed to start within 20s ({self.cwd})")
        if self._session is None:
            raise MCPClientError("MCP session was not established")

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._bootstrap())
            self._loop.run_forever()
        except Exception:
            log.exception("MCP loop crashed for %s", self.cwd)
        finally:
            with suppress(Exception):
                self._loop.run_until_complete(self._teardown())
            self._loop.close()

    async def _bootstrap(self) -> None:
        params = StdioServerParameters(
            command=self.python,
            args=["-m", self.module],
            cwd=str(self.cwd),
            env=self._build_env(),
        )
        self._stdio_ctx = stdio_client(params)
        read, write = await self._stdio_ctx.__aenter__()
        self._session_ctx = ClientSession(read, write)
        self._session = await self._session_ctx.__aenter__()
        await self._session.initialize()
        self._ready.set()

    async def _teardown(self) -> None:
        if self._session_ctx is not None:
            with suppress(Exception):
                await self._session_ctx.__aexit__(None, None, None)
            self._session_ctx = None
            self._session = None
        if self._stdio_ctx is not None:
            with suppress(Exception):
                await self._stdio_ctx.__aexit__(None, None, None)
            self._stdio_ctx = None

    def stop(self) -> None:
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None
        self._loop = None
        self._session = None

    def restart(self) -> None:
        self.stop()
        self.start()

    # ---------- calls ----------

    def _submit(self, coro) -> Any:
        if self._loop is None or self._session is None:
            raise MCPClientError("client not started")
        fut: Future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(timeout=self.call_timeout)
        except Exception as exc:
            self._failures += 1
            if self._failures >= 3:
                log.warning("MCP client %s hit 3 consecutive failures; restarting",
                            self.cwd.name)
                with suppress(Exception):
                    self.restart()
                self._failures = 0
            raise

    def call(self, tool: str, **args: Any) -> Any:
        """Call an MCP tool and return the parsed JSON payload (first text block)."""
        async def _do():
            result = await self._session.call_tool(tool, arguments=args)
            return _extract_payload(result)
        out = self._submit(_do())
        self._failures = 0
        return out

    def health(self) -> bool:
        try:
            async def _do():
                await self._session.list_tools()
                return True
            return bool(self._submit(_do()))
        except Exception:
            return False

    def list_tool_names(self) -> list[str]:
        async def _do():
            r = await self._session.list_tools()
            return [t.name for t in r.tools]
        return self._submit(_do())


def _extract_payload(result: Any) -> Any:
    """MCP `CallToolResult` has a list of `content` blocks; pick the first text/JSON."""
    if getattr(result, "isError", False):
        text = ""
        for block in getattr(result, "content", []) or []:
            text += getattr(block, "text", "") or ""
        raise MCPClientError(f"tool error: {text or 'unknown'}")
    blocks = getattr(result, "content", None) or []
    for block in blocks:
        text = getattr(block, "text", None)
        if text is None:
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured
    return None
