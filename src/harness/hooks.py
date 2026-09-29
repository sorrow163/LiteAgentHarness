import fnmatch
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.core.message import ToolCall
from src.core.tool import CallVerdict, DenySource

SESSION_START = "session_start"
USER_PROMPT_SUBMIT = "user_prompt_submit"
PRE_TOOL_USE = "pre_tool_use"
POST_TOOL_USE = "post_tool_use"
STOP = "stop"

HOOK_EVENTS = (SESSION_START, USER_PROMPT_SUBMIT, PRE_TOOL_USE, POST_TOOL_USE, STOP)

#: 各时机的 payload 形状（由调用点负责填，钩子按这些键读）：
#: - SESSION_START       {"session_id": …, "workspace": …}
#: - USER_PROMPT_SUBMIT  {"message": …}
#: - PRE_TOOL_USE        {"call": ToolCall}
#: - POST_TOOL_USE       {"call": ToolCall, "message": ToolMessage}
#: - STOP                {"stop_reason": …}

@dataclass
class HookResult:
    """一个钩子的裁决。全 ``None`` / ``False`` = "我没意见"。"""

    block: bool = False                 # 要不要拦截
    reason: str = ""                    # 拦截原因
    replace_args: dict[str, Any] = None  # 仅 ``PRE_TOOL_USE`` 有效：改写参数
    add_context: str = ""               # 追加给模型的上下文（会话级时机用）


HookFn = Callable[[dict[str, Any]], HookResult | None]


class HookManager:
    """按时机挂钩子；``fire`` 合并同一时机的所有钩子的裁决。"""

    def __init__(self) -> None:
        # {<Hook Event>: [(matcher, HookFn)]}
        self._hooks: dict[str, list[tuple[str, HookFn]]] = {e: [] for e in HOOK_EVENTS}

    def register(self, event: str, fn: HookFn, matcher: str = "*") -> None:
        """挂一个钩子。``matcher`` 对工具类事件按工具名 ``fnmatch``（如 ``"bash"`` / ``"*_file"``）。"""
        if event not in self._hooks:
            raise ValueError(f"未知钩子时机: {event}（可用: {HOOK_EVENTS}）")
        self._hooks[event].append((matcher, fn))

    def fire(self, event: str, payload: dict[str, Any]) -> HookResult:
        """触发一个时机的所有钩子，合并裁决：

        任何一个 ``block`` 即 ``block``；``replace_args`` 后写覆盖先写；``add_context`` 累积。
        """
        merged = HookResult()
        tool_name = payload.get("call").name if isinstance(payload.get("call"), ToolCall) else None
        for matcher, fn in self._hooks.get(event, ()):
            if tool_name is not None and not fnmatch.fnmatch(tool_name, matcher):
                continue
            hook_result = fn(payload)
            if hook_result is None:
                continue
            if hook_result.block:
                return HookResult(block=True, reason=hook_result.reason or "被钩子拦截")
            if hook_result.replace_args is not None:
                merged.replace_args = hook_result.replace_args
                # 改写后的调用要传给后面的钩子（它们看到的该是"要执行的那个"）
                payload = {**payload, "call": ToolCall(
                    name=payload["call"].name, args=hook_result.replace_args,
                    id=payload["call"].id,
                )}
            if hook_result.add_context:
                merged.add_context += ("\n" if merged.add_context else "") + hook_result.add_context
        return merged

    def judge(self, call: ToolCall, inner=None) -> CallVerdict:
        """把 ``PRE_TOOL_USE`` 编译成**判定链的第一段**：先钩子、再 ``inner``（权限）。

        返回 :class:`CallVerdict` —— 和权限策略同一个形状，所以两者可以串起来，
        而"要不要问人"（``action="ask"``）能一路交到循环手里。

        - 钩子拦截 → ``action="deny"``、``source="hook"``（落成 ``blocked_by_hook``；
          老实现返回 ``str``，会被错记成"策略拒绝"）；
        - 钩子改写参数 → 交给 ``inner`` 的是**改写后的调用**（"审的就是要执行的"），
          放行时也返回改写后的那个（循环据此执行、并写进 ``tool_start``）；
        - 钩子没话说 → 直接返回 ``inner(call)``。
        """
        hook_result = self.fire(PRE_TOOL_USE, {"call": call})

        if hook_result.block:
            return CallVerdict(action="deny", call=call, source=DenySource.DENY_HOOK,
                               reason=hook_result.reason or "被钩子拦截")

        judged_call = call
        if hook_result.replace_args is not None:
            judged_call = ToolCall(name=call.name, args=hook_result.replace_args, id=call.id)

        if inner is None:
            return CallVerdict(action="allow", call=judged_call, source=DenySource.DENY_HOOK, reason="钩子放行")
        return inner(judged_call)