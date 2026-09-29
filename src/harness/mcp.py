import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from typing import Any

from src.core.tool import Tool

PROTOCOL_VERSION = "2025-06-18"


class MCPError(RuntimeError):
    pass


def _mcp_tool_name(server: str, tool: str) -> str:
    """把 ``mcp__<server>__<tool>`` 净化成内核能接受的名字（``^[a-z_][a-z0-9_]*$``）。

    MCP 工具名可能带 ``-``、大写、点号等非法字符；小写 + 非 ``[a-z0-9_]`` 换成 ``_``。
    """
    raw = f"mcp__{server}__{tool}".lower()
    return re.sub(r"[^a-z0-9_]", "_", raw) or "mcp_tool"

def get_system_encoding():
    """动态获取当前系统的终端编码"""
    if sys.platform == 'win32':
        return 'gbk'  # Windows 简体中文默认是 GBK (cp936)
    else:
        return 'utf-8'  # Linux/Mac 默认是 UTF-8

class MCPServerStdio:
    """通过 stdio 与一个 MCP server 子进程通信的最小客户端。"""

    def __init__(self, command: list[str], name: str = "server", timeout_s: float = 30,
                 env: dict[str, str] | None = None) -> None:
        self.command = command
        self.name = name
        self.timeout_s = timeout_s
        #: 追加到子进程环境上的变量（``None`` = 完全继承父进程）。
        #: **合并**而不是替换：MCP server 通常是个解释器/npx，PATH 一类必须留着。
        self.env = env
        self.proc: subprocess.Popen | None = None
        self._id = 0
        self._lock = threading.Lock()          # M5：串行发/收，别让并发调用偷应答
        self._lines: "queue.Queue[str | None]" = queue.Queue()

    # ---- 传输层 ----
    def _send(self, msg: dict[str, Any]) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise MCPError(f"MCP server {self.name!r} 未启动或已关闭")
        self.proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def _pump(self) -> None:
        """读线程：把 server 的 stdout 一行行塞进队列；EOF 时塞一个 ``None`` 哨兵。"""
        try:
            assert self.proc and self.proc.stdout
            for line in self.proc.stdout:
                self._lines.put(line)
        except Exception:
            pass
        self._lines.put(None)

    def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        with self._lock:
            self._id += 1
            self._send({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}})
            deadline = time.monotonic() + self.timeout_s
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.close()
                    raise MCPError(f"MCP server {self.name!r} 超时（{self.timeout_s}s）无响应，已关闭")
                try:
                    line = self._lines.get(timeout=remaining)
                except queue.Empty:
                    self.close()
                    raise MCPError(f"MCP server {self.name!r} 超时（{self.timeout_s}s）无响应，已关闭")
                if line is None:
                    raise MCPError(f"MCP server {self.name!r} 意外退出")
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue                        # server 往 stdout 打了日志，忽略
                if "id" not in msg:
                    continue                        # 通知（progress / log），跳过
                if msg["id"] != self._id:
                    continue                        # 有锁 + 串行时不该发生；防御性跳过
                if "error" in msg:
                    raise MCPError(f"{method} 失败: {msg['error'].get('message')}")
                return msg.get("result")

    def _notify(self, method: str) -> None:
        self._send({"jsonrpc": "2.0", "method": method})

    # ---- 生命周期 ----

    def start(self) -> "MCPServerStdio":
        """启动子进程并完成握手。握手失败会**回收子进程**再抛"""
        self.proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding=get_system_encoding(), errors="replace",
            env=({**os.environ, **self.env} if self.env else None),
        )
        self._reader = threading.Thread(target=self._pump, daemon=True, name=f"mcp-{self.name}")
        self._reader.start()
        try:
            self._request(
                "initialize",
                {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                 "clientInfo": {"name": "LiteAgentHarness", "version": "0.1"}})

            self._notify("notifications/initialized")
        except BaseException:
            self.close()
            raise
        return self

    def close(self) -> None:
        """关停子进程：先 ``terminate`` 再 ``wait``，必要时 ``kill``。幂等。"""
        if self.proc is None:
            return
        proc, self.proc = self.proc, None
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def __enter__(self) -> "MCPServerStdio":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.close()

    # ---- tools 能力 ----

    def list_tools(self) -> list[dict[str, Any]]:
        return self._request("tools/list").get("tools", [])

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        result = self._request("tools/call", {"name": name, "arguments": arguments})
        texts = [c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"]
        out = "\n".join(t for t in texts if t)
        if result.get("isError"):
            raise MCPError(out or "工具执行失败")
        return out

    # ---- 适配：远端工具 → 内核 Tool ----

    def as_tools(self) -> list[Tool]:
        """把 server 的每个工具包装成内核 ``Tool``，命名 ``mcp__<server>__<tool>``
        （与主流 harness 的命名惯例一致，权限规则可按前缀匹配整个 server）。"""
        out: list[Tool] = []
        for spec in self.list_tools():
            tool_name = spec["name"]

            def call(_name=tool_name, **kwargs: Any) -> str:
                return self.call_tool(_name, kwargs)

            description = spec.get("description") or \
                f"远端工具 {tool_name}（MCP server {self.name}）"
            out.append(Tool.from_schema(
                name=_mcp_tool_name(self.name, tool_name),
                description=description,
                parameters=spec.get("inputSchema") or {"type": "object", "properties": {}},
                func=call,
                timeout_s=self.timeout_s + 5,      # 执行器兜底；内部超时会先 kill server
            ))
        return out
