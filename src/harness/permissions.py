import json
from dataclasses import dataclass, field

from src.core.message import ToolCall

from src.core.tool import CallVerdict, DenySource

ALLOW, DENY, ASK = "allow", "deny", "ask"

RISK_READ, RISK_WRITE, RISK_EXEC, RISK_SPAWN = "read", "write", "exec", "spawn"

#: 工具默认风险等级。**查不到的一律按 exec 处理**（fail-safe）。
DEFAULT_RISK: dict[str, str] = {
    "read_file": RISK_READ,
    "list_dir": RISK_READ,
    "glob_files": RISK_READ,
    "grep": RISK_READ,
    "read_skill": RISK_READ,
    "todo_write": RISK_WRITE,
    "write_file": RISK_WRITE,
    "edit_file": RISK_WRITE,
    "task": RISK_SPAWN,  # 子代理：自身无害，危险在它内部的工具
}

#: 各权限模式下，风险等级 → 默认动作。
_MODE_TABLE: dict[str, dict[str, str]] = {
    "readonly": {RISK_READ: ALLOW, RISK_WRITE: DENY, RISK_EXEC: DENY, RISK_SPAWN: ALLOW},
    "ask": {RISK_READ: ALLOW, RISK_WRITE: ASK, RISK_EXEC: ASK, RISK_SPAWN: ALLOW},
    "accept_edits": {RISK_READ: ALLOW, RISK_WRITE: ALLOW, RISK_EXEC: ASK, RISK_SPAWN: ALLOW},
    "yolo": {RISK_READ: ALLOW, RISK_WRITE: ALLOW, RISK_EXEC: ALLOW, RISK_SPAWN: ALLOW},
}

#: 审批超时后按**风险等级**取的默认结果。
#:
#: 为什么不是"一律拒绝"或"一律放行"：
#: 只读操作超时放行的最坏结果是"多读了一个文件"，而写 / 执行 / 委派是不可逆的 ——
#: 一律拒绝会让只读任务在用户不在时寸步难行，一律放行会在用户离开时改他的文件。
#: **方向与 ``as_gate`` 里"没有审批通道就拒绝"一致**：没人回答 = 不放行不可逆操作。
DEFAULT_TIMEOUT_DECISION: dict[str, bool] = {
    RISK_READ: True,
    RISK_WRITE: False,
    RISK_EXEC: False,
    RISK_SPAWN: False,
}


def invocation_summary(call: ToolCall) -> str:
    """把一次调用压成一行可匹配、可展示的字符串，如 ``bash(git status)``。"""
    if call.name == "bash":
        head = str(call.args.get("command", ""))
    elif len(call.args) == 1:
        head = str(next(iter(call.args.values())))
    else:
        head = json.dumps(call.args, ensure_ascii=False)
    return f"{call.name}({head})"


def _invocation_inner(call: ToolCall) -> str:
    """取 ``summary`` 里括号内的部分，如 ``bash(git status)`` → ``git status``。"""
    summary = invocation_summary(call)
    return summary[len(call.name) + 1: -1]


@dataclass(frozen=True)
class AuditEntry:
    """记账的一条：这次调用最后怎么了。

    ``kind`` 区分两类记录：``judge``（判定链的结论）与 ``approval``（问过人的那次）。
    ``by`` 说明"谁做的决定"（``human`` / ``timeout`` / ``none`` / 空 = 没问人）——
    "人拒绝"和"没有人可问"对模型、对调试是完全不同的两件事。
    """

    kind: str  # judge / approval / mode
    invocation: str  # 一行摘要，如 ``write_file(a.py)``
    action: str  # allow / deny / ask
    by: str = ""  # 有问人时才有：human / timeout / none
    reason: str = ""  # 为什么这么判

    def __str__(self) -> str:
        tail = f"（{self.by}: {self.reason}）" if self.by else (f"（{self.reason}）"
                                                              if self.reason else "")
        return f"[{self.kind}] {self.invocation} → {self.action}{tail}"


@dataclass
class PermissionPolicy:
    """权限策略：**模式 ＋ 风险等级表**

    ``known`` 是"当前 harness 实际提供了哪些工具"（构造时由 ``Harness`` 填）。
    **不在 ``known`` 里的一律拒绝** —— 连"这个名字根本不存在"都要拦下来，
    而且**不问人**（问了也执行不了）。``known`` 为空表示不检查（单测省事）。
    """

    mode: str = "ask"

    risk: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_RISK))

    #: 记账账本（内存）。**进程结束就没了**；要长期留痕是另一件事。
    audit: list[AuditEntry] = field(default_factory=list)

    known: frozenset = frozenset()

    #: 审批默认等多久（秒），传给通道。
    ask_timeout_s: float = 60

    #: **方向不可反**：不可逆操作的默认必须是拒绝。
    timeout_decision: dict[str, bool] = field(default_factory=lambda: dict(DEFAULT_TIMEOUT_DECISION))

    def __post_init__(self) -> None:
        # 模式拼错直接报错，不静默降级
        if self.mode not in _MODE_TABLE:
            raise ValueError(f"未知权限模式 {self.mode!r}（可用: {sorted(_MODE_TABLE)}）")

    # ---- 模式（TUI 会改它） ----
    def set_mode(self, mode: str) -> None:
        """换权限模式（**校验过的入口**）。TUI 的模式开关调这个"""

        if mode not in _MODE_TABLE:
            raise ValueError(
                f"未知权限模式 {mode!r}（可用: {sorted(_MODE_TABLE)}）—— 拒绝切换，保持原模式 {self.mode!r}")
        self.mode = mode

    @property
    def modes(self) -> list[str]:
        """可选模式列表（给 TUI 画开关用）。"""
        return sorted(_MODE_TABLE)

    # ---- 判定（纯函数：不阻塞、不问人、不改状态） ----
    def risk_of(self, call: ToolCall) -> str:
        """这次调用的风险等级。**没登记的一律 ``exec``**（fail-safe 1）。"""
        return self.risk.get(call.name, RISK_EXEC)

    def propose(self, call: ToolCall) -> bool:
        """**没有人可问**时该给什么结果：按风险等级取默认值（fail-safe 2）。

        循环用它填 ``ApprovalRequest.proposed``（"倒计时结束后会怎样"），
        也用它当驱动方 ``send(None)`` 时的默认答案。
        """
        return self.timeout_decision.get(self.risk_of(call), False)

    def judge(self, call: ToolCall) -> CallVerdict:
        """给出裁决：``allow`` / ``deny`` / ``ask``。**不问人、不记事件。**"""
        if self.known and call.name not in self.known:
            # 名字不存在 → 直接拒绝，别去问人（问了也执行不了）
            return CallVerdict(action=DENY, call=call, source="unknown",
                               reason=f"没有名为 {call.name!r} 的工具")
        level = self.risk_of(call)
        action = _MODE_TABLE[self.mode].get(level, ASK)
        return CallVerdict(action=action, call=call, source=DenySource.DENY_POLICY, risk=level,
                           reason=f"{self.mode} 模式下 {level} 级工具默认 {action}")

    def record(self, *, kind: str, call: ToolCall, action: str, by: str = "", reason: str = "") -> None:
        """记一条账（``Agent.audit`` 就接在这里）。"""
        self.audit.append(AuditEntry(kind=kind, invocation=invocation_summary(call),
                                     action=action, by=by, reason=reason))
