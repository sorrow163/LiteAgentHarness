import fnmatch
import os
import re
import tempfile
from itertools import islice
from pathlib import Path

from src.core.tool import Tool, tool

#: ``read_file`` 单次最多读这么多行（防止一次读几万行把上下文打爆）
MAX_READ_LINES = 500

#: ``grep`` 跳过超过这个大小的文件（兆级文件不该被逐行扫）
GREP_MAX_FILE_BYTES = 1024 * 1024

#: 每次 ``bash`` 从子进程读一块的字节数（边读边限流，别全量进内存）
_READ_CHUNK = 8192


def make_coding_tools(root: str) -> list[Tool]:
    """构造一套锁定在 ``root`` 工作区的编码工具。

       工具拿运行时环境靠**闭包**：所有路径都相对 ``root`` 解析。
       """

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    def _resolve(path: str) -> str:
        """路径 jail：任何解析结果越出 root 一律拒绝。"""
        target = (root / path).resolve()


        if target != root and not str(target).startswith(str(root) + os.sep):
            raise PermissionError(f"路径越出工作区: {path!r}")

        return str(target)

    def _atomic_write(path: str, content: str) -> None:
        """同目录写临时文件，再 ``os.replace`` 原子换名"""
        directory = os.path.dirname(path) or "."
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".writing")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
            os.replace(tmp, path)  # 同一文件系统上的原子改名
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @tool
    def read_file(path: str, offset: int = 0, limit: int = 200) -> str:
        """读取工作区内的文本文件，返回带行号的内容。
        path: 相对工作区的路径。offset/limit: 从第 offset 行（从 0 起）最多读 limit 行，
        大文件请分段读取而不是一次读完。"""
        if offset < 0:
            raise ValueError(f"offset 不能是负数，收到 {offset}")
        limit = max(1, min(int(limit), MAX_READ_LINES))
        with open(_resolve(path), "r", encoding="utf-8", errors="replace") as handle:
            window = list(islice(handle, offset, offset + limit))
            has_more = bool(handle.readline())
        body = "".join(f"{i + offset + 1:>5}\t{line}" for i, line in enumerate(window))
        tail = "\n…（后面还有多行）" if has_more else ""
        return (body or "(空文件)") + tail

    @tool
    def write_file(path: str, content: str) -> str:
        """在工作区内创建或整体覆盖一个文件。仅用于新建文件或全量重写；
        修改既有文件请优先用 edit_file。"""
        target = _resolve(path)
        os.makedirs(os.path.dirname(target) or root, exist_ok=True)
        _atomic_write(target, content)
        return f"已写入 {path}（{len(content)} 字符）"

    @tool
    def edit_file(path: str, old: str, new: str) -> str:
        """对既有文件做精确替换：old 必须与文件中某段内容**完全一致且唯一**。
        找不到或出现多处都会报错——请提供更长的上下文片段来唯一定位。"""
        target = _resolve(path)
        with open(target, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        count = text.count(old)
        if count == 0:
            raise ValueError(f"未找到要替换的内容（old 与文件不一致）: {old[:80]!r}")
        if count > 1:
            raise ValueError(f"old 在文件中出现 {count} 次，无法唯一定位，请扩大片段")
        _atomic_write(target, text.replace(old, new, 1))
        return f"已编辑 {path}"

    @tool
    def list_dir(path: str = ".") -> str:
        """列出工作区内某目录的内容（目录以 / 结尾）。"""
        target = _resolve(path)
        entries = sorted(os.listdir(target))
        out = [name + "/" if os.path.isdir(os.path.join(target, name)) else name
               for name in entries]
        return "\n".join(out) or "(空目录)"

    @tool
    def glob_files(pattern: str) -> str:
        """按通配符递归查找文件名，如 "**/*.py"、"src/*.md"。返回相对路径列表（一律正斜杠）。"""
        pattern = pattern.replace("\\", "/")  # 模型偶尔给反斜杠，统一成正斜杠
        hits: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", "node_modules")]
            for filename in filenames:
                rel = os.path.relpath(os.path.join(dirpath, filename), root).replace(os.sep, "/")
                if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(filename, pattern):
                    hits.append(rel)
        return "\n".join(sorted(hits)[:200]) or "(无匹配)"

    @tool
    def grep(pattern: str, path: str = ".", max_matches: int = 50) -> str:
        """在工作区内按正则搜索文件内容，返回 "文件:行号:内容"。
        pattern: Python 正则。path: 限定搜索的子目录，默认全工作区。"""
        rx = re.compile(pattern)
        base = _resolve(path)
        hits: list[str] = []
        limit_reached = False
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", "node_modules")]
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                # M9：跳过超大文件，别把一个 2GB 的东西逐行扫一遍
                try:
                    if os.path.getsize(filepath) > GREP_MAX_FILE_BYTES:
                        continue
                    with open(filepath, "r", encoding="utf-8") as handle:
                        for lineno, line in enumerate(handle, 1):
                            if rx.search(line):
                                rel = os.path.relpath(filepath, root).replace(os.sep, "/")
                                hits.append(f"{rel}:{lineno}:{line.rstrip()}")
                                if len(hits) >= max_matches:
                                    limit_reached = True
                                    break
                except (UnicodeDecodeError, OSError):
                    continue
                if limit_reached:
                    break
            if limit_reached:
                break
        return "\n".join(hits) or "(无匹配)"

    @tool
    def todo_write(content: str) -> str:
        """维护你的任务清单（覆盖式写入）。做多步任务时，先把计划写进来，
        每完成一步就更新状态——这份清单会帮你在长任务中保持方向。"""
        _atomic_write(os.path.join(root, ".agent_todo.md"), content)
        return "任务清单已更新"

    tools = [read_file, write_file, edit_file, list_dir, glob_files, grep, todo_write]

    return tools
