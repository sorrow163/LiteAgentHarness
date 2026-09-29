# -*- coding: utf-8 -*-
"""启动前把控制台切到 UTF-8 —— 否则界面画不出来。

## 为什么必须做

中文 Windows 上 Python 的 ``sys.stdout.encoding`` 是 **GBK**（cp936），终端也按 cp936
解码。界面上要画的字符里，有一批**不在 GBK 里**（半块 ``▐``、勾号 ``✔``、项目符号 ``•``）
—— 往 GBK 的流里写一个就抛 ``UnicodeEncodeError``，整个界面当场崩掉。

这不是假想的风险：这套 harness 真跑基准时就栽在同一个坑上 —— 负责画界面的那个库往 GBK
控制台写 ``•``（U+2022）抛 ``UnicodeEncodeError``，任务明明满分、进程却以退出码 1 结束
（当时是靠给子进程设 ``PYTHONIOENCODING=utf-8`` 绕过去的）。那次要的是"输出不崩"，
这次要的是"界面能画"。

## 两件事要一起做

1. **控制台代码页**：终端按什么**解码**我们写出去的字节；
2. **Python 的流编码**：我们按什么**编码**写出去。

只改一样都不行：只改 1 → Python 仍按 GBK 编码、写不出的字符照样抛异常；
只改 2 → Python 吐 UTF-8 字节、终端按 cp936 解成乱码（中文全花）。

**不用 `chcp`**：那要起一个子进程，还得解析它的输出。``SetConsoleOutputCP`` /
``SetConsoleCP`` 是同一件事的 API 版，直接问得到"改之前是什么"，退出时能还原。
没有控制台（重定向、CI）时这两个 API 返回 0，此时**什么都不做** —— 界面前提本来
就是能交互的终端，硬改反而会污染重定向出去的文件。
"""
from __future__ import annotations

from collections.abc import Callable

#: UTF-8 的代码页号（Windows 上就是它）。
UTF8_CODE_PAGE = 65001


def ensure_utf8_console() -> Callable[[], None]:
    """把控制台与标准流切到 UTF-8，返回**还原函数**（在应用退出时调）。

    还原是为了不把用户终端留在 65001 上：本程序跑完，他接着用的还是原来那个代码页。
    """
    restore_code_page = _switch_code_page(UTF8_CODE_PAGE)
    restore_streams = _switch_streams()
    if restore_code_page is None and restore_streams is None:
        return lambda: None

    def restore() -> None:
        if restore_streams is not None:
            restore_streams()
        if restore_code_page is not None:
            restore_code_page()

    return restore


def _switch_code_page(code_page: int) -> Callable[[], None] | None:
    """切控制台代码页，返回"切回去"的函数；没有控制台或切不动则返回 ``None``。"""
    try:
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    except Exception:                        # noqa: BLE001  非 Windows / 取不到库：不做
        return None

    old_output = kernel32.GetConsoleOutputCP()
    old_input = kernel32.GetConsoleCP()
    if not old_output and not old_input:      # 没有控制台（重定向 / CI）
        return None
    if old_output == code_page and old_input == code_page:
        return None

    changed = False
    if old_output and old_output != code_page:
        kernel32.SetConsoleOutputCP(code_page)
        changed = True
    if old_input and old_input != code_page:
        kernel32.SetConsoleCP(code_page)
        changed = True
    if not changed:
        return None

    def restore() -> None:
        if old_output:
            kernel32.SetConsoleOutputCP(old_output)
        if old_input:
            kernel32.SetConsoleCP(old_input)

    return restore


def _switch_streams() -> Callable[[], None] | None:
    """把三个标准流的编码切成 UTF-8，返回"切回去"的函数。

    ``errors="replace"``：真遇到这个字体/编码里没有的字符时**画成一个替代符**，
    而不是把整个界面崩掉。界面是"看得见就行"的东西，不值当为一个大不了的字陪你死。
    """
    import sys

    saved: list[tuple[object, str, str]] = []
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        encoding = getattr(stream, "encoding", None)
        errors = getattr(stream, "errors", None)
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:                    # noqa: BLE001  切不动就算了（比如已经被包了一层）
            continue
        saved.append((stream, encoding, errors))
    if not saved:
        return None

    def restore() -> None:
        for stream, encoding, errors in saved:
            try:
                stream.reconfigure(encoding=encoding, errors=errors)   # type: ignore[attr-defined]
            except Exception:                # noqa: BLE001
                pass

    return restore
