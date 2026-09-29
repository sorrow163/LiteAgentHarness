import contextvars
from dataclasses import dataclass


@dataclass
class RunContext:
    """当前 run 的信息"""

    run_id: str = ""
    session_span: str | None = None      # 主会话 span（子代理的 run_start 挂在它下面）
    workspace: str | None = None
    result_dir: str | None = None
    memory: str = ""

@dataclass
class CallContext:
    """当前工具调用的信息（由执行器在 ``tool.invoke`` 前后设置）。"""

    call_id: str = ""
    name: str = ""

_run: contextvars.ContextVar = contextvars.ContextVar("harness_run_context", default=None)
_call: contextvars.ContextVar = contextvars.ContextVar("harness_call_context", default=None)


# 工具函数执行时上下文变量管理
def set_call(ctx: CallContext) -> contextvars.Token:
    return _call.set(ctx)

def get_call() -> CallContext:
    ctx = _call.get()
    return ctx if ctx is not None else CallContext()

def reset_call(token) -> None:
    _call.reset(token)

# 子代理执行时的上下文变量管理
def get_run() -> RunContext:
    ctx = _run.get()
    return ctx if ctx is not None else RunContext()

def set_run(ctx: RunContext) -> contextvars.Token:
    return _run.set(ctx)

def reset_run(token) -> None:
    _run.reset(token)

