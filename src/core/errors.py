
class LiteAgentHarnessError(Exception):
    """内核所有异常的基类。"""


class ToolBindError(LiteAgentHarnessError):
    """工具定义有问题 —— 在**绑定期**就报出来，而不是等运行时才发现。"""


class Cancelled(LiteAgentHarnessError):
    """协作式取消：工具在长循环里调用 ``ctx.cancel.check()`` 时主动抛出。

    执行器捕获它 → 产出 ``outcome=cancelled`` 并**闭合 span**
    """
class SessionError(LiteAgentHarnessError):
    """会话异常"""

class SandboxError(LiteAgentHarnessError):
    """沙箱层的基类。"""


class SandboxUnavailable(SandboxError):
    """沙箱用不了（挂不上授权 / 造不出令牌 / 起不了进程）—— **命令没有运行**。

    和"命令跑了但失败"是两回事：前者该去修环境，后者该去改命令。工具必须把它们
    分开报，否则模型会盯着自己的命令改半天。
    """
