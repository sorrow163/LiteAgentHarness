import os
from collections.abc import Iterable, Sequence, Iterator, Generator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.core.context import ContextManager, load_memory_files, memory_source_paths
from src.core.errors import SessionError
from src.core.events import Event, EventType, DataKey, get_new_id
from src.core.loop import Agent
from src.core.message import ToolCall
from src.core.models import BaseChatModel
from src.core.tool import Tool, CallVerdict
from src.core.toolkit import make_coding_tools
from src.harness.hooks import HookManager, SESSION_START, USER_PROMPT_SUBMIT, POST_TOOL_USE, STOP
from src.harness.mcp import MCPServerStdio
from src.harness.permissions import PermissionPolicy, DEFAULT_TIMEOUT_DECISION
from src.harness.sandbox.bash import (ENFORCEMENT, EXECUTOR, Sandbox, SandboxConfig,
                                      make_sandbox_tool, workspace_sid_of)
from src.harness.session import workspace_key, SessionStore, Session
from src.harness.skills import load_skills, make_skill_tool, skills_section
from src.harness.subagent import SubagentDef, make_task_tool

REPO_ROOT = Path(__file__).resolve().parents[2]

#: 默认会话仓库：``<仓库根>/sessions/`` —— **算成绝对路径**。
#: 刻意不给相对路径：相对路径一旦按工作区解析，工作区里就会长出一份 sessions。要换位置就显式传 ``session_root``。
DEFAULT_SESSION_ROOT = str(REPO_ROOT / "sessions")


def sandbox_meta(sandbox: SandboxConfig | None) -> dict[str, Any]:
    """沙箱配置 → 记进 ``head.meta.config`` 的那一小段。

    **执行侧不读它**（loop / 工具 / 权限层都不看），它是写给**事后**读 trace 的人与程序的：
    轨迹可比性取决于"跑在什么环境里"——同一份工作区在不同环境下行为不同，`workspace`
    字段看不出这个；沙箱开与不开更是两种完全不同的轨迹形状。

    所以这里**整段照抄**沙箱配置，不挑字段：挑字段才是过度设计，以后加沙箱开关还得回来
    改这里。另外三个**派生**出来的事实也要记，它们不在配置里但决定了当时的环境：

    - ``executor``：一眼分辨"这批是不是带沙箱跑的"；
    - ``enforcement``：**隔离强度**。这一档是 ``partial``（只挡写、读不受限、
      `Everyone` 必须留在限制列表里），**不许**写成 `full` —— 事后比较两条轨迹时，
      "沙箱"这两个字底下到底是什么强度，比"开没开"更容易被误读；
    - ``workspace_sid``：当时授的到底是哪一条 ACE。有了它，事后可以照着去 `icacls` 查。

    **沙箱实例的身份不记**（比如私有临时目录）：它一次会话一个样，而且会话结束就删了；
    能力 SID 是**纯函数**，要的话自己能算，所以记它是记"当时用的那个值"，不是记实例。
    """
    if sandbox is None:
        return {"executor": "host"}
    return {"executor": EXECUTOR, "enforcement": ENFORCEMENT,
            "workspace_sid": workspace_sid_of(sandbox.workspace),
            **asdict(sandbox)}


def answer_from(events: Iterable[Event]) -> str:
    """从**事件流**里取这次 run 的最终回答：最后一条**没有 tool_calls** 的 ``assistant_message``。
    """
    text = ""
    for event in events:
        if event.type == EventType.ASSISTANT_MESSAGE:
            message = event.data.get(DataKey.MESSAGE)
            if message is not None and not getattr(message, "tool_calls", None):
                text = message.content or ""
    return text


def session_root_path(session_root: str) -> str:
    """把 ``session_root`` 定成一个**绝对路径**（会话文件存**程序侧**，与工作区无关）。

    - 绝对路径 → 照用（``normpath`` 去尾斜杠）；
    - 相对路径 → 相对**仓库根**（``REPO_ROOT``）解析。

    **刻意不接收 ``workspace``**：以前相对路径是按工作区解析的，于是工作区里会长出一份
    ``.harness/sessions/``（"数据跟着被试对象走"，删工作区就丢轨迹）。改成语义之后
    工作区不再是参数 —— 留一个用不上的参数只会让人以为"它跟工作区有关"。

    存量会话：以前默认落在 ``<workspace>/.harness/sessions/``。改默认值**不迁移**它们 ——
    想继续用旧目录，显式传 ``session_root`` 即可（映射按目录扫，换个目录就换一批会话）。
    注意这**只是换目录**，不会自动接上旧会话 —— 要接着某条聊，得用 ``open(session_id)``。
    """
    if os.path.isabs(session_root):
        return os.path.normpath(session_root)
    return os.path.normpath(os.path.join(REPO_ROOT, session_root))


DEFAULT_SYSTEM_PROMPT = """\
你是一个严谨的编码助手，在一个受限的工作区内帮用户完成任务。
- 动手前先看清现状（读文件/列目录），不臆测文件内容。
- 多步任务先用 todo_write 列计划，完成一步更新一步。
- 修改文件优先用 edit_file 做精确替换；只有新建/全量重写才用 write_file。
- 工具被拒绝时不要原样重试，向用户说明并寻求替代方案。
- 回答简洁，用中文。"""


@dataclass
class HarnessConfig:
    model: BaseChatModel  # 必需：一个 BaseChatModel 实例
    workspace: str = "./workspace"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    #: 权限模式：``readonly`` / ``ask``（默认）/ ``accept_edits`` / ``yolo``。
    #: **TUI 上随时可改**：``harness.policy.set_mode("yolo")``（走校验过的入口）。
    permission_mode: str = "ask"
    #: 审批超时（秒）：塞进 ``ApprovalRequest``，给弹窗画倒计时用。驱动方可以自行缩短。
    approval_timeout_s: float | None = None
    #: 没有人可问时按风险等级取的默认结果（``{}`` ＝ 用出厂值：只读放行，其余拒绝）。
    timeout_decision: dict[str, bool] = field(default_factory=dict)
    skills_dir: str | None = None
    mcp_servers: Sequence[MCPServerStdio] = ()
    subagents: Sequence[SubagentDef] = ()
    extra_tools: Sequence[Tool] = ()
    session_root: str = DEFAULT_SESSION_ROOT
    max_turns: int = 40
    max_context_tokens: int = 60_000
    compact_threshold: float = 0.75
    keep_recent: int = 8
    #: 给不给**宿主侧的编码工具**（``read_file`` / ``write_file`` / ``edit_file`` /
    #: ``list_dir`` / ``glob_files`` / ``grep`` / ``todo_write``）。
    #:
    #: 关掉它是给"工作区不在宿主上"的场景用的（容器里跑基准任务、远程环境）：那套工具是按
    #: **宿主路径**解析的，在那种场景里模型改的是宿主上一个无关目录，而它自己看不出来 ——
    #: 两边不一致比"少几个工具"危险得多。关掉之后工具面**完全由 ``extra_tools`` 决定**。
    coding_tools: bool = True
    #: 沙箱：给了就在能力面里多一个 ``sandbox_bash``。``None``（默认）= 整套机制不存在。
    #:
    #: 这一档是**宿主内写受限令牌**（只挡写，不挡读），沙箱身份是**工作区**而不是会话 ——
    #: 所以一个 `Harness` 一个沙箱，跨会话原样留着（见 `_attach_sandbox`）。
    sandbox: SandboxConfig | None = None
    summarizer: BaseChatModel | None = None
    #: 测试 / 高级用法可直接注入；不给就用默认的。
    #: **策略不从这里注入** —— 它由本文件按上面几个字段构造（模式 ＋ 风险表 ＋ 超时语义），
    #: 这样"哪边生效"不会有第二种答案。
    hooks: HookManager | None = None
    meta: dict[str, Any] | None = None


class Harness:
    """一个可复用的 agent 运行器：会话、能力面、控制面、上下文面都在这装配好。"""

    def __init__(self, cfg: HarnessConfig) -> None:
        # **工作区先规范化，再干别的**：工作区身份在两处被用到 —— 映射（`head.meta.workspace`）
        # 与工具层的路径解析（`make_coding_tools` / 子代理 / `run`）。如果只规范化映射那一侧，
        # 同一个工作区就会有两个写法在系统里游走（日志、报错、将来的容器名各取一个）。
        # 归一之后 `cfg.workspace` 本身就是键，`_default_meta` 里不必再算一次。
        cfg.workspace = workspace_key(cfg.workspace)
        # 沙箱配置里那份工作区**必须跟着一起归一**：容器名和挂载都从它派生，两处各留一个
        # 写法就会让同一个工作区在系统里有两个身份（容器名、挂载、元信息各取一个）。
        # **必须在 `_default_meta()` 之前** —— 它读的就是这个字段，晚一步元信息里就会
        # 留下一份没归一的路径，而元信息正是事后区分"跑在什么环境里"的唯一依据。
        if cfg.sandbox is not None:
            cfg.sandbox.workspace = cfg.workspace
        self.cfg = cfg
        self.hooks = cfg.hooks or HookManager()
        self.skills = load_skills(cfg.skills_dir) if cfg.skills_dir else []
        self.store = SessionStore(session_root_path(cfg.session_root))
        self.meta = cfg.meta or self._default_meta()
        self._session: Session | None = None
        self._session_span: str | None = None
        self._agent: Agent | None = None
        self._mcp: list[MCPServerStdio] = []
        self._mcp_started: list[MCPServerStdio] = []

        # 沙箱：**懒装配在 `open()` 里**（那时才有会话、才确定要用它），但**一个 Harness 只配
        # 一个**，跨会话原样留着 —— 这一档的身份是工作区，而工作区在 Harness 存续期内不会变
        # （`open()` 拒绝跨工作区）。详见 `_attach_sandbox()`。
        # 这里只留两个槽位；`None` 时 `_build_tools` 就不会加 `sandbox_bash`。
        # （工作区已经在函数开头对齐过了 —— 见那里的注释。）
        self.sandbox: Sandbox | None = None
        self._sandbox_tool: Tool | None = None

        # 权限策略：**模式 ＋ 风险等级表**。
        # 它只回答"能不能跑 / 要不要问人"（纯函数）；问人由循环 yield 请求、驱动方 send 决定。
        self.policy = PermissionPolicy(
            mode=cfg.permission_mode,
            # None / {} = 不做覆盖，用出厂值（安全默认不可反）
            ask_timeout_s=(cfg.approval_timeout_s if cfg.approval_timeout_s is not None else 60),
            timeout_decision=(dict(cfg.timeout_decision) if cfg.timeout_decision
                              else dict(DEFAULT_TIMEOUT_DECISION)))

        # 把"当前实际提供哪些工具"交给策略（不在里面的名字一律拒绝 —— 模型的幻觉工具名
        # 不该走到审批去问人）。为此能力面要装配两次：这里一次（只为拿名字），
        # `_build_agent` 再装一次（真正交给 Agent）。所以 `known` 是**建完工具后回填**的。
        self.policy.known = frozenset(t.name for t in self._build_tools())

    def _default_meta(self) -> dict[str, Any]:
        return {
            "agent": "main",
            # `cfg.workspace` 在 `__init__` 里已经规范化过（见那里的注释），所以这里直接记。
            "workspace": self.cfg.workspace,
            "model": getattr(self.cfg.model, "model", type(self.cfg.model).__name__),
            "provider": getattr(self.cfg.model, "provider", "unknown"),
            "harness": "LiteAgentHarness@0.1.0",
            "config": {"permission_mode": self.cfg.permission_mode, "parallel_tools": False,
                       "max_turns": self.cfg.max_turns,
                       "max_context_tokens": self.cfg.max_context_tokens,
                       "session_root": self.cfg.session_root,
                       # 跑在什么执行器上 —— 见 `sandbox_meta()` 的说明：它是评估可比性的前提
                       "sandbox": sandbox_meta(self.cfg.sandbox)},
        }

    def _build_tools(self) -> list[Tool]:
        """装配能力面（工具清单）。

        会被调用两次，这是刻意的：``__init__`` 一次（只为把"存在哪些工具"交给权限策略
        —— 未知工具名一律拒绝），``_build_agent`` 一次（真正交给 Agent，那时 MCP 已 start）。
        **两份必须一致**，所以 ``_build_agent`` 末尾会用最终那份刷新 ``policy.known``。

        MCP 的 ``server.start()`` 不在这里（它只该跑一次，由 ``_build_agent`` 负责）；
        这里只取已经 start 过的 server 的工具表。
        """
        coding = make_coding_tools(self.cfg.workspace) if self.cfg.coding_tools else []
        tools: list[Tool] = coding + list(self.cfg.extra_tools)
        if self._sandbox_tool is not None:
            tools.append(self._sandbox_tool)
        if self.skills:
            tools.append(make_skill_tool(self.skills))
        for server in self._mcp:
            tools += server.as_tools()
        if self.cfg.subagents:
            tools.append(make_task_tool(
                list(self.cfg.subagents), self.cfg.model, store=self.store,
                parent_session_id=self._session.id if self._session else "",
                meta=self.meta, judge=self.judge, summarizer=self.cfg.summarizer))
        return tools

    def judge(self, call: ToolCall) -> CallVerdict:
        """**判定链：先钩子、再权限**（钩子可改写参数，权限必须裁决"改写后"的那次调用）。

        纯函数：不阻塞、不问人、不改状态 —— "要不要问人"作为 ``action="ask"`` **交出去**，
        由循环决定怎么问（yield 请求 / 让驱动方 send 决定）。
        """
        return self.hooks.judge(call, inner=self.policy.judge)

    def inject(self, text: str) -> None:
        """中途插一条用户消息（steering）：下一轮模型调用前生效、并落成 ``user_message`` 事件。"""
        if self._agent is not None:
            self._agent.inject(text)

    # ---- 外部控制（TUI 交互） ----
    def interrupt(self) -> None:
        """请求停止当前 run（协作式：在下一个消息边界生效）。TUI 按 Esc 就调它。"""
        if self._agent is not None:
            self._agent.interrupt()

    def _audit(self, *, kind: str, call: ToolCall, action: str, by: str = "",
               reason: str = "") -> None:
        """记账出口（塞给 ``Agent.audit``）：判定与审批的结论都进 ``policy.audit``。

        审批结论**不进事件流**（控制面的动作；评估跑在权限全开下不会发生），
        所以这里是"刚才为什么被拒 / 谁批的"的唯一记录。
        """
        self.policy.record(kind=kind, call=call, action=action, by=by, reason=reason)

    def list_sessions(self) -> list[str]:
        """**当前工作区**的主控会话 id，按"最近活动"倒序（新 → 旧），不含子代理的 ``.sub-N``。

        顺序跟 ``list_session_summaries()`` 一致（同一个 ``find_main``），够 TUI 直接拿来做
        "最近会话"列表。**要接着某条聊，把它的 id 传给 ``open(session_id)``** ——
        ``open()`` 不带 id 是**新建**，不会自动接上最近那条。
        """
        return [head.session_id for head, _ in self.store.find_main(workspace_key(self.cfg.workspace))]

    def all_sessions(self) -> list[str]:
        """仓库里**所有**工作区的主控会话 id。"""
        return self.store.list()

    def child_sessions(self, session_id: str) -> list[str]:
        """某个主控会话的**子代理**附属文件 id（``<main>.sub-N``，同目录并列）。

        "查看子代理内容"以"打开了这个主控"为前提 —— 所以入口在这里，而不是在会话列表里。
        """
        return self.store.child_ids(session_id)

    def _check_workspace(self, session: Session) -> None:
        """打开的会话必须属于当前工作区（不一致就明确失败，不静默修正）。"""
        stored = session.meta.get("workspace")
        current = workspace_key(self.cfg.workspace)
        if stored != current:
            raise SessionError(
                f"会话 {session.id!r} 属于别的工作区，拒绝打开：\n"
                f"  会话记录的工作区: {stored}\n"
                f"  当前工作区:       {current}\n"
                f"（会话文件存在程序侧，可以跨工作区打开；但工具按**当前**工作区解析路径，"
                f"混着用会在错的项目上动手。要接着跑就切到那个工作区，或显式传 session_root 隔离。）"
            )

    @staticmethod
    def _session_span_from(session: Session) -> str:
        events = session.events()
        if events and events[0].type == EventType.SESSION_START:
            return events[0].span
        return get_new_id()

    def session_summary(self, session_id: str) -> dict[str, Any]:
        """读一个会话的元信息（给会话列表页 / 状态栏）。"""
        session = self.store.open(session_id)
        events = session.events()
        last = events[-1] if events else None
        return {
            "id": session.id,
            "parent": session.parent,  # -1 主控 / 否则父会话（子代理的 .sub-N 文件）
            "agent": session.meta.get("agent"),
            "model": session.meta.get("model"),
            "workspace": session.meta.get("workspace"),
            "closed": bool(events) and events[-1].type == EventType.SESSION_END,
            "last_ts": last.ts if last else None,
            "message_count": len(session.messages),
        }

    def list_session_summaries(self) -> list[dict[str, Any]]:
        """会话列表页要的摘要：**当前工作区**的主控，按"最近活动"倒序（新 → 旧）。

        子代理的 ``.sub-N`` 不在这里，用 ``child_sessions(main_id)`` 单独取 ——
        "查看子代理内容"以"打开了这个主控"为前提。
        """
        heads = self.store.find_main(workspace_key(self.cfg.workspace))
        return [self.session_summary(head.session_id) for head, _ in heads]

    def _build_agent(self, session_add_context: str) -> None:
        extra = skills_section(self.skills)
        if session_add_context:
            extra = (extra + "\n\n" if extra else "") + session_add_context
        context = ContextManager(
            max_context_tokens=self.cfg.max_context_tokens,
            compact_threshold=self.cfg.compact_threshold,
            keep_recent=self.cfg.keep_recent,
            extra_context=extra,
        )

        for server in self.cfg.mcp_servers:  # MCP 只 start 一次（__init__ 装工具时还没 start）
            if server not in self._mcp_started:
                server.start()
                self._mcp_started.append(server)
            if server not in self._mcp:
                self._mcp.append(server)
        tools = self._build_tools()
        # 刷新策略眼里的"存在哪些工具"：这里才是**权威**的一份。
        # `__init__` 那次装配早于 MCP start，拿不到 MCP 工具；而 `known` 少一个名字的后果是
        # "这个工具永远被拒（名字不存在）"—— 所以必须在工具真正定稿的这一刻同步一次。
        self.policy.known = frozenset(t.name for t in tools)

        self._agent = Agent(
            model=self.cfg.model, tools=tools, system_prompt=self.cfg.system_prompt,
            max_turns=self.cfg.max_turns, context=context,
            judge=self.judge, propose=self.policy.propose, audit=self._audit,
            summarizer=self.cfg.summarizer)

    # ---- 生命周期 ----
    def open(self, session_id: str | None = None) -> Session:
        """打开一个会话并装配好能力面（工具 + agent）。两种种情形：

        1. 给了 id，会判定文件是否存在，当前工作区是不是对应上；
        2. 没给 id，则是新建会话
        """
        opened = True
        if session_id is not None:
            session = self.store.open(session_id)
            self._check_workspace(session)

        else:
            session = self.store.create(meta=self.meta)
            opened = False

        if not opened:
            result = self.hooks.fire(SESSION_START, {"session_id": session.id, "workspace": self.cfg.workspace})
            if result.block:
                raise Exception(f"会话被钩子拒绝: {result.reason}")
            session_add_context = result.add_context
        else:
            session_add_context = ""

        self._session = session
        self._session_span = self._session_span_from(session)
        if not opened:
            session.write_event(Event(type=EventType.SESSION_START, span=self._session_span))
        self._attach_sandbox()
        self._build_agent(session_add_context)
        return session

    def _attach_sandbox(self) -> None:
        """给这个 `Harness` 配一个沙箱 —— **一个 Harness 一个，跨会话原样留着**。
        """
        if self.sandbox is not None or self.cfg.sandbox is None:
            return
        self.sandbox = Sandbox(self.cfg.sandbox)
        self._sandbox_tool = make_sandbox_tool(self.sandbox)

    def close(self) -> None:
        """关会话：收沙箱、写 ``session_end``、fire ``STOP``、关掉所有 MCP 子进程。"""
        # **收沙箱放最前面**：`Sandbox.close()` 保证不抛，而后面那些（钩子、MCP 子进程）
        # 都可能抛 —— 一个收不掉的沙箱不该有权把后续清理跳过。
        # 沙箱是懒建的（第一次真跑命令才造令牌），没建过就是 no-op；收掉后也丢掉，
        # 下次 `open()` 重新装配。
        if self.sandbox is not None:
            self.sandbox.close()
            self.sandbox = None
            self._sandbox_tool = None
        if self._session is not None and self._session_span is not None:
            self._session.write_event(Event(type=EventType.SESSION_END,
                                            span=self._session_span, data={"reason": "closed"}))
        for server in self._mcp:
            server.close()
        self._mcp = []
        self._agent = None
        self._session = None
        self._session_span = None

    # ---- 主入口 ----
    def run_stream(self, prompt: str, session_id: str | None = None) -> Generator[Event]:
        """跑一次 run，**逐条 yield 事件**；落盘非惰性（每条在 yield 之前已写进会话文件）。

        **它是唯一的执行入口，而且必须"双向驱动"**：需要审批时它会 yield 一条
        ``approval_request``（live-only），驱动方用 ``gen.send(决定)`` 把答案送回来 ——
        决定直达循环里那个挂起点，不需要线程、不需要信箱。
        """
        if self._agent is None or (session_id is not None and session_id != self._session.id):
            self.open(session_id)

        session:Session = self._session
        prior = session.messages          # 最新的上下文窗口 = 多轮记忆

        # 项目记忆每 run 装配一次（AGENTS.md 等，C3 的修法）。
        # 路径单独取一份给 `sys_prompt` 事件用：内容在下面那个字符串里，而**出处只能靠路径**。
        memory = load_memory_files(self.cfg.workspace)
        memory_sources = memory_source_paths(self.cfg.workspace)

        # USER_PROMPT_SUBMIT 钩子：拦截 / 注入
        result = self.hooks.fire(USER_PROMPT_SUBMIT, {"prompt": prompt})
        if result.block:
            raise Exception(f"输入被钩子拒绝: {result.reason}")
        if result.add_context:
            memory = (memory + "\n\n" if memory else "") + "[hook 注入]\n" + result.add_context

        pending_calls: dict[str, ToolCall] = {}
        inner:Generator = self._agent.run(prompt, prior_messages=prior,
                                session_span=self._session_span, memory=memory,
                                memory_sources=memory_sources,
                                workspace=self.cfg.workspace)
        # ★ **双向透传**：不能写 `for event in inner: ...; yield event` —— 那样驱动方
        #   `send` 的值会在这里被吃掉（审批点永远收到 None ＝ 没人回答）。
        #   `yield from` 才能把 send 的值送到循环里那个审批挂起点。
        answer = None
        while True:
            try:
                event = inner.send(answer) if answer is not None else next(inner)
            except StopIteration:
                return
            if event.type == EventType.TOOL_START:
                pending_calls[event.span] = ToolCall(
                    name=event.data["name"], args=dict(event.data["args"]),
                    id=event.data[DataKey.CALL_ID])
            elif event.type == EventType.TOOL_RESULT:
                call = pending_calls.pop(event.span, ToolCall(name="?", args={}, id="?"))
                self.hooks.fire(POST_TOOL_USE, {"call": call, "message": event.data.get(DataKey.MESSAGE)})
            elif event.type == EventType.RUN_END:
                self.hooks.fire(STOP, {"stop_reason": event.data.get(DataKey.STOP_REASON)})
            session.write_event(event)    # 先落盘，再 yield —— 消费者崩了也不丢
            answer = yield event          # ← 消费者 send 回来的东西（审批决定）继续往上传