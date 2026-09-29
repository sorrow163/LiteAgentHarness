from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from src.core.errors import SandboxUnavailable
from src.core.message import ToolMessage
from src.core.tool import Tool, tool
from src.harness.sandbox.restricted_token import (ConfinedRun, RestrictedTokenSandbox,
                                                  workspace_sid)

#: `head.meta.config.sandbox.executor` 的取值。名字里带上**机制**，因为强度不一样：
#: 写受限令牌只挡写，与容器那种"读写都拒 + 断网"不是一回事，事后读 trace 的人必须看得出差别。
EXECUTOR = "win-write-restricted-token"

#: 这一档的隔离强度。**不许写 `"full"`** —— 官方定性就是 partial（`Everyone` 必须留在限制
#: SID 列表里、硬链接会别名化、读不受限），写成 full 是自欺。
ENFORCEMENT = "partial"

#: ★ **退出码保真包装器** —— 不是美化，是必需。
#:
#: PowerShell 5.1 的 `-Command` **不把原生命令的退出码传出来**。本机实测：
#:
#:     -Command "cmd /c exit 7"                    → 1   ✗ 精确码丢了
#:     -Command "cmd /c exit 3; Write-Output after" → 0   ✗✗ 失败被后面成功的语句盖掉
#:     -Command "python -c \"print(1)\""             → 1   ✗ 实际是 0xC0000409（WindowsApps 的桩）
#:
#: 第二条最危险：**"命令失败了"会变成"成功"**。而工具契约（结尾一定有 `[exit code: N]`）
#: 就架在这个数字上 —— 数字不真，模型分不出"测试没过"与"跑成功了"，也看不出"命令崩了"。
#: 包上之后以上三条分别得到 `7` / `3` / `3221226505`（= 0xC0000409），实测 18/18
#: 通过，证据在 `temp/probe_ps_exit_code.py`。
#:
#: 三个细节缺一不可：**先清零**（否则可能撞上陈旧值）、**跑完立刻存 `$?`**（`if` 的条件求值
#: 自己会改写 `$?`）、**原生退出码优先**（`$?` 只有真/假，会把 7 压成 1）。
#:
#: ⚠ **它救不了什么**（本机实测，证据 `temp/probe_ps_script_exit.py`）：失败如果发生在
#: **被调用的 `.ps1` 脚本内部**，退出码仍然是 `0` —— 那是 PowerShell 的标准语义（脚本的
#: 退出码由脚本作者负责，`-File` 也一样），不是包装器能修的：
#:
#:     命令文本里直接失败        → 1  ✓（包装器管用）
#:     & '内在失败的.ps1'        → 0  ✗（PowerShell 语义）
#:     脚本里自己写 exit 1       → 1  ✓
#:     $ErrorActionPreference='Stop' 后再调用 → 1 ✓（能救，但改变了语言的语义，我们不做）
#:
#: 对照：`.bat` / `.cmd` 那条**没有**这个问题 —— cmd.exe 会设自己的退出码，PowerShell 把它
#: 记进 `$LASTEXITCODE`，包装器就能拿到。所以这条限制只针对 PowerShell 脚本。
#: 缓解办法是**告诉模型**（见 `tool_description`：脚本内部的失败可能不体现在退出码上，看 stderr）。
EXIT_CODE_WRAPPER = (
    "$LASTEXITCODE = 0\n"
    "{command}\n\n"
    "$ok = $?\n"
    "if ($LASTEXITCODE -ne 0) {{ exit $LASTEXITCODE }}\n"
    "if (-not $ok) {{ exit 1 }}\n"
    "exit 0\n"
)


def default_shell_argv() -> tuple[str, ...]:
    """默认 shell 的 argv 前缀（命令字符串拼在最后）。

    Windows 上按**存在性**挑，而不是硬写一个路径：`SystemRoot` 被改过的机器上那个绝对路径
    不存在，而 PATH 上的 `powershell.exe` 还在。最后兜底 `cmd.exe /c`（它一定有，虽然能力最弱
    —— 真走到那一步要记得把 `exit_code_wrapper` 设成 `None`，cmd 自己就保真）。

    POSIX 上给 `sh -c`：**这一档的后端只有 Windows**（`available()` 会明说），但函数本身
    不该在别的平台上给出一个不存在的 `powershell.exe` —— 那会让"默认值"变成一句谎话。
    """
    if sys.platform != "win32":
        return ("/bin/sh", "-c")
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    candidates = (
        (os.path.join(system_root, "System32", "WindowsPowerShell", "v1.0", "powershell.exe"),
         "-NoProfile", "-NonInteractive", "-Command"),
        ("powershell.exe", "-NoProfile", "-NonInteractive", "-Command"),
        (os.path.join(system_root, "System32", "cmd.exe"), "/c"),
    )
    for argv in candidates:
        # 绝对路径要真的存在；PATH 上的名字交给系统去解析（那是它的事）。
        if not os.path.isabs(argv[0]) or os.path.isfile(argv[0]):
            return argv
    return candidates[-1]


def default_exit_code_wrapper() -> str | None:
    """默认的退出码包装器（**与 `default_shell_argv()` 是一对**）。

    POSIX 的 `sh -c` 自己就把退出码传出来，不需要包；Windows PowerShell 5.1 需要
    —— 理由见 `EXIT_CODE_WRAPPER` 里的实测证据。
    """
    return EXIT_CODE_WRAPPER if sys.platform == "win32" else None


def workspace_sid_of(workspace: str) -> str:
    """工作区 → 能力 SID 的文本形式。**按 `realpath` 之后的路径算**。
    """
    return workspace_sid(os.path.realpath(workspace))


def interpreter_for(workspace: str) -> str:
    """这条沙箱里"跑 Python"该用哪个解释器：**先看工作区自己的 `.venv`**。

    用户要的是"命令跑在与工作区同一个 venv 里"，而 harness 自己可能跑在**另一个**解释器里
    （全局安装、`uv` 的临时环境、IDE 挑的那个都可能）。所以先认工作区里的 `.venv`，
    没有才退回 `sys.executable`（harness 自己那个）。
    """
    #: 第二种形状是给 Linux 留的（项目目标是 Windows + Linux 都要），现在还没有后端用它。
    for relative in ((".venv", "Scripts", "python.exe"), (".venv", "bin", "python")):
        candidate = Path(workspace).joinpath(*relative)
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def child_env(interpreter: str) -> dict[str, str]:
    """沙箱子进程要追加/覆盖的环境：把解释器所在目录塞到 `PATH` 最前面（见模块文档）。

    `VIRTUAL_ENV` 只在"解释器确实在 venv 里"时才设 —— 目录名不是 `Scripts` / `bin` 就说明
    那是个普通安装（`sys.executable` 的兜底情形），这时设它反而是在骗工具。
    """
    scripts = os.path.dirname(interpreter)
    if not scripts:
        return {}
    env = {"PATH": scripts + os.pathsep + os.environ.get("PATH", "")}
    if os.path.basename(scripts).lower() in ("scripts", "bin"):
        env["VIRTUAL_ENV"] = os.path.dirname(scripts)
    return env


@dataclass
class SandboxConfig:
    """沙箱的全部可调项"""

    #: 宿主侧工作区（绝对路径）
    workspace: str
    #: 命令超时的默认值与上限（秒）。超时后杀的是**整棵进程树**。
    default_timeout_s: float = 60.0
    max_timeout_s: float = 600.0
    #: 交给执行器的结果长度上限（超了会落盘，并在结尾告诉模型去哪读全文）。
    max_result_chars: int = 8_000
    #: 沙箱自己收输出的上限（字节）。与 `max_result_chars` **不是一回事**：那个是"给模型看
    #: 多少"，这个是"最多留多少在内存里"，两个分开设是为了各自独立可调。
    max_output_bytes: int = 200_000
    #: 子进程的资源上限（Job Object）。`None` = 那一项不限。
    memory_mb: int | None = 2048
    active_process_limit: int | None = 256
    #: 私有临时目录的父目录；`None` = 用 `%TEMP%`。子进程的 `TEMP`/`TMP` 会被指到
    #: 它下面的一个会话私有目录（DSH 就是这么做的）。
    temp_root: str | None = None
    #: 收尾撤不撤工作区授权。**默认不撤**，两个理由：
    #:
    #: 1. **它是跨会话的缓存**：留着，`_has_exact_ace` 会认出它并跳过 —— 而"跳过"省掉的不只是
    #:    一次写，是**整个继承传播**（`SetNamedSecurityInfoW` 会把可继承 ACE 传播到目标下整个
    #:    层级，那才是真正花时间的一步）。DSH 就是这么做的：本会话现场那条工作区 ACE 正是它
    #:    上一次留下来的，我们直接复用。（早期版本是"自己递归遍历全树逐个挂 ACE"，那时留着
    #:    的好处还没这么明显；现在也不吃亏。）
    #: 2. **它不构成越权**：能力 SID 由工作区路径派生，是非唯一授权、账号库里没有主体能解析
    #:    到它；要用它得先造一个把它放进限制列表的令牌 —— 那已经是"你这个用户自己的权限"了。
    #:
    #: 设成 `True` 是给"用完就扔的临时工作区"用的（那种场景残留没有意义）。
    revoke_on_close: bool = False
    #: 命令交给谁解释（argv 前缀）。
    shell_argv: tuple[str, ...] = field(default_factory=default_shell_argv)
    #: 把命令包一层以确保退出码为真。**与 `shell_argv` 是一对**：换 shell 时一起换
    #: （`cmd.exe` 与 POSIX 的 `sh -c` 自己就保真，设成 `None`）。模板里 `{command}` 会被
    #: 替换成模型写的那条命令。
    exit_code_wrapper: str | None = field(default_factory=default_exit_code_wrapper)


class Sandbox:
    """一条命令通道：**懒建令牌、跑命令、收尾**。别的都不干。

    - **懒建**：`__init__` 只算路径（便宜），第一次真跑命令时才 `ensure()`（造令牌 ＋ 挂 ACE）
      —— 不用沙箱的会话一分钱不付。挂 ACE 要遍历整棵工作区树，这是这一档最贵的一步。
    - **`close()` 收尾**：撤不撤授权看配置（默认不撤，见 `SandboxConfig.revoke_on_close`）；
      私有临时目录**总是**删（它本来就是一次性的）。由 `Harness.close()` 调。

    **不做的事**：不装东西、不碰宿主进程、不判断命令该不该跑（那是权限层的事）。
    """

    def __init__(self, cfg: SandboxConfig) -> None:
        self.cfg = cfg
        #: 命令里 `python` 会解析到的解释器（"同一个 venv"就是它），也写进工具说明。
        self.interpreter = interpreter_for(cfg.workspace)
        # ★ 后端自己会 `resolve()`（realpath）—— 那一步正是能力 SID 与 DSH 对得上的关键：
        # `cfg.workspace` 是 `workspace_key()` 归一过的（Windows 上 `normcase` 会把盘符小写化），
        # 而 realpath 会把文件系统里的**真实大小写**找回来。见 `workspace_sid()` 的说明。
        self._box = RestrictedTokenSandbox(
            cfg.workspace, temp_root=cfg.temp_root, memory_mb=cfg.memory_mb,
            active_process_limit=cfg.active_process_limit,
            revoke_on_close=cfg.revoke_on_close,
            extra_env=child_env(self.interpreter),
            max_output_bytes=cfg.max_output_bytes)

    # ---- 只读的小窗口 ----
    @property
    def notes(self) -> list[str]:
        """后端记下的备注（授权统计、私有临时目录、失败原因……）。装配与排障时读它。"""
        return self._box.notes

    @property
    def temp_dir(self) -> str | None:
        """私有临时目录（`ensure()` 之后才有）。"""
        return self._box.temp_dir

    @property
    def workspace_sid(self) -> str:
        """工作区能力 SID 的文本形式（**纯函数**，与 `ensure()` 之后会用的那个是同一个值）。"""
        return workspace_sid(self._box.workspace)

    # ---- 生命周期 ----
    def ensure(self) -> str:
        """把令牌与授权建起来（**幂等**）。返回工作区能力 SID 的文本形式。"""
        return self._box.ensure()

    def close(self) -> None:
        """收尾。**绝不抛**（它挂在会话收尾路径上）。"""
        self._box.close()

    # ---- 跑命令 ----
    def run(self, command: str, *, timeout_s: float | None = None) -> ConfinedRun:
        """跑一条命令（交给 `shell_argv` 解释）。**超时也返回已经读到的输出。**

        `cwd` 不在这里传：后端默认就把子进程的 cwd 设成工作区（两处是同一个路径，
        分两处写只会让"到底哪个生效"变成一个需要查的问题）。
        """
        text = command
        if self.cfg.exit_code_wrapper is not None:
            text = self.cfg.exit_code_wrapper.format(command=command)

        return self._box.run([*self.cfg.shell_argv, text], timeout_s=self._clamp(timeout_s))

    def _clamp(self, timeout_s: float | None) -> float:
        """超时归一到 `[1, max_timeout_s]`。**下限 1 秒**：0 或负数会让子进程刚起来就被杀，
        那种失败看起来像"命令有问题"，实际是参数问题。"""
        want = self.cfg.default_timeout_s if timeout_s is None else float(timeout_s)
        return max(1.0, min(want, self.cfg.max_timeout_s))


def _render(result: ConfinedRun) -> str | ToolMessage:
    """一次运行的记录 → 给模型看的一段字。

    **退出码单独一行**（`[exit code: N]`）：模型要能一眼分出"跑完了但失败"。
    超时是另一回事 —— 命令**没跑完**，所以按错误报（`is_error=True`），并把已经拿到的输出
    一起带上（命令挂住时，那半截输出是唯一能说明"它卡在哪"的线索）。
    """
    parts: list[str] = []
    if result.stdout.strip():
        parts.append(result.stdout.rstrip("\n"))
    if result.stderr.strip():
        parts.append("[stderr]\n" + result.stderr.rstrip("\n"))
    if result.truncated:
        parts.append("[输出超过上限，尾部已丢弃]")
    body = "\n".join(parts) if parts else "(无输出)"

    if result.timed_out:
        return ToolMessage(
            content=f"{body}\n[sandbox] 命令超时（整棵进程树已被杀掉），没有跑完；"
                    f"上面是它被杀之前已经产生的输出",
            is_error=True)
    return f"{body}\n[exit code: {result.return_code}]"


def tool_description(sandbox: Sandbox) -> str:
    """`sandbox_bash` 的接口文档（**写给模型看的**，所以讲的是"怎么用"而不是"怎么实现"）。

    单独一个函数而不是写成内层函数的 docstring：这里要**插进真实的解释器路径**，
    而 f-string 不是字符串常量，挂不到 `__doc__` 上（那样工具描述会变成空）。
    """
    return f"""在隔离的沙箱里执行一条命令，返回它的输出和退出码。

命令交给 **Windows PowerShell 5.1** 解释（不是 bash）：
- **没有 `&&`**（那是 PowerShell 7 的）。要连着跑就用 `;`，或者分成两次调用；
- 管道 `|`、`$(...)`、`2>&1`、`>` / `>>` 都能用；
- `python` / `pytest` / `pip` 已经指向工作区那个 venv（`{sandbox.interpreter}`），直接写就行。

环境（不用试探，按这个来就行）：
- 工作目录是**工作区根** —— 在这里读写文件就是直接改你的工作区；
- **写**只允许工作区和一个沙箱私有的临时目录；**往别的地方写会被系统拒绝**
  （典型报错是 `PermissionError: [Errno 13]` 或"拒绝访问"）。
  ★ 那不是你的命令写错了，是边界在起作用 —— 别改命令反复重试；
- **读不受限**：宿主机上你这个用户能读的东西它都能读；
- **有网**：装包、拉代码都能跑通；
- 内存与进程数有上限，跑飞了会被终止；超时后**整棵进程树**会被杀掉。

结果怎么读：
- 结尾一定有 `[exit code: N]`；非零就是命令自己失败了，照着改命令；
- ★ **例外：`.ps1` 脚本内部的失败可能不体现在退出码上。** PowerShell 的规矩是"脚本的退出码
  由脚本作者负责"，所以 `& '你的脚本.ps1'` 里某个 cmdlet 失败时，退出码**可能仍是 0**。
  要判断成败，**看 stderr**（报错会写在那里），或者在你写的脚本里显式 `exit 1`。
  （`.bat` / `.cmd` 没有这个问题。）
- 结尾是 `[sandbox] …` 时，那是**沙箱环境的问题、不是你的命令写错了** ——
  不要改命令反复重试，把这句话告诉用户。"""


def make_sandbox_tool(sandbox: Sandbox) -> Tool:
    """造 `sandbox_bash` 工具。

    **只收一个 `Sandbox`**：工作区、解释器、shell 方言、超时上限全在它身上，工具不需要
    再知道"现在跑的是哪条会话" —— 这一档的沙箱是**工作区级**的，不跟会话走（见模块文档）。
    """

    @tool(name="sandbox_bash",
          description=tool_description(sandbox),
          parameters_desc={
              "command": "要在沙箱里执行的命令，交给 Windows PowerShell 5.1 解释。"
                         "**没有 `&&`**，要连着跑用 `;`。",
              "timeout_ms": "超时（毫秒）。默认 60000，超过上限时按上限算；"
                            "超时后整棵进程树会被杀掉。",
          },
          max_result_chars=sandbox.cfg.max_result_chars)
    def sandbox_bash(command: str, timeout_ms: int = 60_000) -> str | ToolMessage:
        #: 接口文档在 `tool_description()` 里（那段要插真实路径，写不成 docstring）。
        try:
            result = sandbox.run(command, timeout_s=timeout_ms / 1000.0)
        except SandboxUnavailable as exc:
            # **沙箱起不来 ≠ 命令写错了**：分开报，否则模型会盯着自己的命令改半天。
            return ToolMessage(content=f"[sandbox] 命令没有运行：{exc}", is_error=True)
        return _render(result)

    return sandbox_bash
