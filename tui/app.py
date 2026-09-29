# -*- coding: utf-8 -*-
"""把桥、控件、按键装成一个应用。

## 数据流

``输入框 ──▶ RunBridge（工作线程）──▶ Harness.run_stream ──▶ 事件``

事件回到界面线程只有一条路：:meth:`HarnessTui._from_worker` 把回调包一层
``call_from_thread``。**界面线程里只改控件，工作线程里只跑内核** —— 两边不许互相摸。

## 三处刻意的取舍

- **用户消息不在本地回显**：它由内核的 ``user_message`` 事件画出来（那是它进历史的
  那一刻）。本地回显看着更"跟手"，但一旦内核在发出它之前就失败了（比如钩子拦下输入），
  界面上就会留着一条**根本没进历史**的消息 —— 界面说的话必须和会话文件一致。
  唯一的例外是**插话**：见 :meth:`HarnessTui.on_input_submitted`。
- **流式文本先攒后画**：一条回答有几百条 ``text_delta``，每条都改一次控件等于每毫秒
  重排一次。攒够一个刷新周期（:data:`FLUSH_INTERVAL_S`）再一次性写上去。
- **状态栏的 token 有账**：本轮之内用 ``assistant_message`` 的用量累加（够实时），
  一轮结束时**改用那条 ``run_end`` 的权威总量**（它含压缩摘要与子代理的消耗）。
  两个来源混着加会重复计数，所以分开两个累加器：:attr:`HarnessTui._closed_usage`
  与 :attr:`HarnessTui._live_usage`。
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Input

from src.core.context import estimate_tokens
from src.core.events import ApprovalDecision, DataKey, Event, EventType, now_ms
from src.core.message import Usage
from src.harness.harness import Harness
from src.harness.session import Session

from tui import render
from tui.bridge import RunBridge
from tui.render import StatusState
from tui.widgets import (
    ApprovalAnswer, ApprovalModal, SessionTreeModal, StatusBar, Transcript)

#: 流式文本的刷新周期（秒）。20 帧/秒：比这更密就只是白烧 CPU（肉眼分不出来），
#: 比这更疏就能看出"一顿一顿"。
FLUSH_INTERVAL_S = 0.05

#: 状态栏的心跳周期（秒）：刷新"本轮耗时"。
TICK_INTERVAL_S = 0.1

IDLE_PLACEHOLDER = ("说点什么，回车发送（Esc 打断 · Ctrl+T 会话树 · "
                    "Shift+Tab 换权限模式 · Ctrl+Q 退出）")
BUSY_PLACEHOLDER = "运行中 —— 回车把这句话插进去（下一轮模型调用前生效）"


class HarnessTui(App[None]):
    """`Harness` 的终端界面。"""

    TITLE = "LiteAgentHarness"

    CSS = """
    Screen { layout: vertical; }
    #transcript { height: 1fr; padding: 0 1; }
    .block { margin-bottom: 1; }
    /* 用户消息：**整块底色带 ＋ 左侧一条粗竖线 ＋ 上下留白** —— 让"我说的那句"一眼能扫到。
       终端里**没有字号可改**（字体由终端与用户决定），能用的杠杆只有底色／边框／留白／粗细这几样；
       底色是"区块级"的差别，比换前景色（跟着主题走、对比度不可控）和加粗（很多字体里几乎看不出）
       都稳。底色刻意用 `$primary-darken-3` 而**不是状态栏那档 `$panel`**：两者贴在一起，同色会糊成一体。 */
    .block.user {
        background: $surface;
        border-left: thick $accent;
        padding: 0 1;
        margin: 1 0;
    }
    .block.reasoning { color: $text-muted; }
    .block.notice { color: $text-muted; }
    .block.error { color: $error; }
    /* 助手回答：Markdown 渲染，给它一点左边距，和工具块区分开 */
    .block.assistant { margin: 0 0 1 0; }
    /* 工具块：头两行 ＋ 可折叠正文。正文左边一条细线，读起来像"这块是它的输出" */
    .block.tool { height: auto; }
    .tool-head { height: auto; }
    .tool-hint { height: auto; }
    .tool-body { margin: 0; border-left: solid $panel-lighten-2; padding-left: 1; }
    /* 可折叠的块：鼠标悬上去或焦点在它身上时给一点底色 —— 不然"能点"这件事看不出来。
       折叠记号本身就是"能展开"的提示，底色只负责回答"点哪儿"。 */
    .foldable:hover, .foldable:focus { background: $boost; }
    #status { height: 1; padding: 0 1; background: $panel; color: $text; text-wrap: nowrap; }
    #prompt { border: tall $accent; }
    ApprovalModal { align: center middle; }
    #approval-box {
        width: 80; max-width: 95%; height: auto;
        border: thick $warning; padding: 1 2; background: $surface;
    }
    #approval-head { text-style: bold; color: $warning; margin-bottom: 1; }
    .approval-fact { height: auto; }
    #approval-count { color: $warning; margin-top: 1; }
    #approval-buttons { height: auto; align-horizontal: right; }
    #approval-buttons Button { margin-left: 2; }
    /* 会话树与子会话视图：都是"盖在当前会话上的一层"，看完就退回原处 */
    SessionTreeModal, SubSessionModal { align: center middle; }
    #tree-box, #view-box {
        width: 90%; max-width: 96%; height: 85%;
        border: thick $accent; padding: 1 2; background: $surface;
    }
    #tree-head, #view-head { height: auto; margin-bottom: 1; }
    #tree-foot, #view-foot { height: auto; margin-top: 1; }
    #tree { height: 1fr; }
    #view-transcript { height: 1fr; padding: 0 1; }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "退出"),
        Binding("ctrl+c", "quit", "退出", show=False, priority=True),
        Binding("escape", "interrupt", "打断"),
        Binding("ctrl+t", "session_tree", "会话树"),
        Binding("shift+tab", "cycle_mode", "权限模式", priority=True),
    ]

    def __init__(self, harness: Harness, session: Session, *,
                 fold_chars: int = render.FOLD_CHARS) -> None:
        super().__init__()
        self.harness = harness
        self.session = session
        #: 思考块的折叠阈值（`config.yml` 的 ``ui.fold_chars``）：超过这么多字符就折起来。
        #: **工具结果不走它** —— 那边无论长短，正文都默认整块折起来（只给一行尺寸提示），
        #: 两种内容的"太长"本来就不是一回事：思考是连续文字，结果是几十行文件内容或日志。
        self.fold_chars = fold_chars
        self.state = StatusState(
            mode=harness.policy.mode,
            model=str(getattr(harness.cfg.model, "model", "?")),
            session=session.id,
            workspace=harness.cfg.workspace,
        )
        # 用完的 run 的权威总量（每条 `run_end.usage`）＋ 正在跑的这一轮的增量估算。
        self._closed_usage = Usage()
        self._live_usage = Usage()
        #: **累计**计数：轮 = 模型调用轮数（每条 `run_end.turn` 是那一次 run 用掉的轮数），
        #: 步 = 工具调用次数（每来一条 `tool_start` 加一）。两者都跨 run 累加 ——
        #: 会话接着聊时"这一轮的第几轮"会从 1 重数，而"累计跑了多少"才是要看的那个数。
        self._turns_done = 0
        self._live_turn = 0
        self._steps = 0
        self._failed = False
        # 正在流式写入的两块（助手文本 / 思考），以及"还没画上去"的缓冲。
        self._stream_block: Any = None
        self._stream_text = ""
        self._stream_buffer = ""
        self._reason_block: Any = None
        self._reason_text = ""
        self._reason_buffer = ""
        # span → (块, 名字, 参数, 是不是子代理)：工具结束时就地重画那一块。
        self._tools: dict[str, tuple[Any, str, dict[str, Any], bool]] = {}
        self._pending_injects: list[str] = []
        self._approval_index: int | None = None
        self._bridge = RunBridge(
            harness,
            on_event=self._from_worker(self._on_event),
            on_approval=self._from_worker(self._on_approval),
            on_answered=self._from_worker(self._on_answered),
            on_error=self._from_worker(self._on_error),
            on_finish=self._from_worker(self._on_finish),
        )

    # ------------------------------------------------------------------
    # 装配
    # ------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Transcript(id="transcript")
        yield StatusBar("", id="status")
        yield Input(placeholder=IDLE_PLACEHOLDER, id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        # ★ 回放读的是**整份事件文件**，不是 `load_ui_events()` 那三类内容事件：
        #   工具调用的**参数**在 `tool_start` 里，少了它，回放出来的工具行只能写成 `read · ?`。
        #   参数本来就在文件里，没理由丢（见 `_replay`）。
        events = self.session.events()
        if any(event.type in render.REPLAY_TYPES for event in events):
            self._replay(events)
            # 划一条线：上面是从会话文件里读回来的，下面是这一次现场跑的。
            self._transcript().add(render.divider_block("以上是历史"), render.KIND_NOTICE)
        self._add_notice(
            f"会话 {self.session.id} · 工作区 {self.harness.cfg.workspace} · "
            f"权限模式 {self.harness.policy.mode}")
        # 能力面**是"配了才有"的**（技能要目录、子代理要注册、沙箱要开关），所以开屏就把它摆出来：
        # 不摆的话，用户只能靠"试一个工具看有没有"来发现哪一档没接上。
        self._add_notice(
            f"能力面：{len(self.harness.policy.known)} 个工具 · "
            f"沙箱{'开' if self.harness.cfg.sandbox else '关'} · "
            f"技能 {len(self.harness.skills)} 个 · "
            f"子代理 {len(self.harness.cfg.subagents)} 个 · "
            f"摘要模型{'有' if self.harness.cfg.summarizer else '无'}")
        self._add_notice("Enter 发送；运行中 Enter 插话、Esc 打断；"
                         "Ctrl+T 看子会话树；Shift+Tab 换权限模式")
        self._refresh_status()
        self.query_one("#prompt", Input).focus()
        self.set_interval(TICK_INTERVAL_S, self._tick)

    def _from_worker(self, callback):
        """把"工作线程里的回调"送回界面线程执行。

        界面已经关掉时 ``call_from_thread`` 会失败（没有事件循环可投递了）—— 那是正常
        退出路径，**丢掉**即可：工作线程是 daemon，随后一起收。
        """

        def wrapped(*args: Any) -> None:
            try:
                self.call_from_thread(callback, *args)
            except Exception:                       # noqa: BLE001
                pass

        return wrapped

    # ------------------------------------------------------------------
    # 输入
    # ------------------------------------------------------------------
    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if self._bridge.running:
            # 运行中：这句话不是"新的一轮"，是**插话**（steering）—— 内核在下一轮模型调用
            # 之前取走它，并落成一条 `user_message`。这里立刻画出来，等那条事件回来时
            # 按文本认领并跳过（否则同一句话会出现两次）。
            self._bridge.inject(text)
            self._pending_injects.append(text)
            self._transcript().add(render.user_block(text, injected=True), render.KIND_USER)
            self._add_notice("已插话：下一轮模型调用前生效")
            return
        self._start_run(text)

    def _start_run(self, prompt: str) -> None:
        self._failed = False
        self._live_usage = Usage()
        self._live_turn = 0                     # 新的一轮 run：这一轮的轮次从 0 起，累计的不动
        self.state.running = True
        self.state.phase = render.PHASE_MODEL
        self._set_input_mode()
        self._refresh_status()
        try:
            self._bridge.start(prompt)
        except Exception as exc:                    # noqa: BLE001  起不来就当一次出错
            self._on_error(exc)

    # ------------------------------------------------------------------
    # 按键
    # ------------------------------------------------------------------
    def action_interrupt(self) -> None:
        """Esc：请求停止当前 run（协作式，内核在下一个消息边界生效）。"""
        if not self._bridge.running:
            return
        self._bridge.interrupt()
        self._add_notice("已请求打断：内核在下一个消息边界停下（不会撕裂历史）")

    def action_cycle_mode(self) -> None:
        """Shift+Tab：在四种权限模式之间轮转。"""
        modes = self.harness.policy.modes
        current = self.harness.policy.mode
        nxt = modes[(modes.index(current) + 1) % len(modes)] if current in modes else modes[0]
        self.harness.policy.set_mode(nxt)
        self.state.mode = nxt
        self._add_notice(f"权限模式 → {nxt}（{_MODE_HINT.get(nxt, '')}）")
        self._refresh_status()

    def action_quit(self) -> None:
        # 退出前先请内核停：不然工作线程还在一半，会话文件会在一个奇怪的时刻收尾。
        if self._bridge.running:
            self._bridge.interrupt()
        self.exit()

    def action_session_tree(self) -> None:
        """``Ctrl+T``：把当前会话**派出去的子子孙孙**摊成一棵树。

        入口放在这儿（而不是"时间线上那一行 ``（子会话 …）`` 点进去"）是因为那是"凭线索找"：
        委派一多，滚回去找那一行很烦。树是总览，一眼看得出这个会话派过几个、各自在干嘛。

        打开的是**只读**视图（见 :class:`tui.widgets.SubSessionModal`）：看子会话是读那条
        ``.sub-N.jsonl``，不是"打开它接着聊" —— 后者会往那条委派记录里写，等于改证据。
        """
        self.push_screen(SessionTreeModal(
            self.harness.store, self.session.id, fold_chars=self.fold_chars,
            live=self.is_running))

    def is_running(self) -> bool:
        """现在有一轮在飞吗。**会话树靠它区分"正在进行"与"未收尾"**（见 `tui/sessions.py`）。"""
        return self._bridge.running

    # ------------------------------------------------------------------
    # 工作线程回过来的四件事（都在界面线程里执行）
    # ------------------------------------------------------------------
    def _on_event(self, event: Event) -> None:
        kind = event.type
        if kind == EventType.TEXT_DELTA:
            self._buffer_stream(event.data.get("text", ""), reasoning=False)
            return
        if kind == EventType.REASONING:
            self._buffer_stream(event.data.get("text", ""), reasoning=True)
            return

        if kind == EventType.MODEL_START:
            self.state.phase = render.PHASE_MODEL
            # 这一轮 run 走到了第几轮（`run_end` 会给权威的总轮数，那时再并入累计）
            self._live_turn = event.turn or self._live_turn
        elif kind == EventType.USER_MESSAGE:
            self._on_user_message(event)
        elif kind == EventType.ASSISTANT_MESSAGE:
            self._on_assistant_message(event)
        elif kind == EventType.TOOL_START:
            self._on_tool_start(event)
        elif kind == EventType.TOOL_RESULT:
            self._on_tool_result(event)
        elif kind == EventType.RUN_END:
            self._on_run_end(event)
        elif kind == EventType.COMPACTION:
            self._transcript().add(render.compaction_block(event), render.KIND_NOTICE)
        elif kind == EventType.ERROR:
            self._failed = True
            self._add_error(event)
        self._refresh_status()

    def _on_user_message(self, event: Event) -> None:
        message = event.data.get(DataKey.MESSAGE)
        text = str(getattr(message, "content", "") or "")
        if text and text in self._pending_injects:
            self._pending_injects.remove(text)      # 插话那一会儿已经画过了
            return
        if text:
            self._transcript().add(render.user_block(text), render.KIND_USER)
        self.state.phase = render.PHASE_MODEL

    def _on_assistant_message(self, event: Event) -> None:
        self._flush_stream()
        message = event.data.get(DataKey.MESSAGE)
        usage = getattr(message, "usage", None)
        if isinstance(usage, Usage):
            self._live_usage = self._live_usage + usage
        text = str(getattr(message, "content", "") or "")
        # 有增量的话文字已经在流式那块里了；没有（非流式模型）才补一块。
        # 两者都走 Markdown 渲染。
        if text and not self._stream_text:
            self._transcript().add_markdown(text, render.KIND_ASSISTANT)
        self._reset_stream()

    def _on_tool_start(self, event: Event) -> None:
        line = render.tool_start_line(event)
        # 工具块一开始就建成：此刻只有"正在调"那一行，没有正文（没什么可折）。
        # 记的是**块 ＋ 名字 ＋ 参数**：块用来就地重画，名字与参数用来重画开头那行
        # （顺手把临时的"运行中…"去掉）—— 回放那条路也是这么配对的。
        view = render.ToolView(line, "", "", "", False)
        block = self._transcript().add_tool(view, render.KIND_TOOL)
        self._tools[str(event.span)] = (
            block, str(event.data.get("name", "?")), dict(event.data.get("args") or {}),
            event.data.get(DataKey.TOOL_TYPE) == "subagent")
        self._steps += 1                        # 一次工具调用 = 一步
        self.state.phase = render.PHASE_TOOL

    def _on_tool_result(self, event: Event) -> None:
        block, name, args, subagent = self._tools.pop(str(event.span), (None, "", {}, False))
        view = render.tool_view(event, name=name, args=args, subagent=subagent)
        if block is None:
            self._transcript().add_tool(view, render.KIND_TOOL)
        else:
            block.update_view(view)

    def _on_run_end(self, event: Event) -> None:
        self._transcript().add(render.run_end_block(event), render.KIND_NOTICE)
        usage = event.data.get(DataKey.USAGE)
        if isinstance(usage, Usage):
            # ★ 换成权威的那一份：`run_end.usage` 含压缩摘要与子代理的消耗，
            #   本轮之内那些"逐条累加"的估算到此作废（两个都加会重复计数）。
            self._closed_usage = self._closed_usage + usage
            self._live_usage = Usage()
        # `run_end.turn` 就是这次 run 用掉的轮数 → 并进累计，这一轮的临时计数归零
        self._turns_done += event.turn or 0
        self._live_turn = 0

    def _on_approval(self, request: Any, index: int) -> None:
        self.state.phase = render.PHASE_APPROVAL
        self._approval_index = index
        self._refresh_status()
        self.push_screen(ApprovalModal(request, index), self._on_modal_closed)

    def _on_modal_closed(self, answer: ApprovalAnswer | None) -> None:
        """弹窗关掉了。``None`` ＝ 被程序收掉的（超时那条路），不是人答的。"""
        if answer is None:
            return
        self._bridge.answer(answer.index, answer.decision)
        word = "批准" if answer.decision == ApprovalDecision.APPROVED else "拒绝"
        self._add_notice(f"审批：{word}")

    def _on_answered(self, index: int, by: str) -> None:
        """桥那边定了（人答了 / 超时了）。**超时要把还开着的弹窗收掉**。"""
        if self._approval_index != index:
            return
        self._approval_index = None
        if by == "timeout":
            if isinstance(self.screen, ApprovalModal) and self.screen.index == index:
                self.pop_screen()
            self._add_notice("审批：等不到回答 —— 按风险等级的默认值继续")

    def _on_error(self, exc: BaseException) -> None:
        self._failed = True
        self._add_notice(f"这一轮没能跑起来：{type(exc).__name__}: {exc}")

    def _on_finish(self) -> None:
        self._flush_stream()
        self._reset_stream()
        self.state.running = False
        self.state.phase = render.PHASE_DONE if self._failed else render.PHASE_IDLE
        if self._pending_injects:
            # 插话比这一轮晚到：它躺在内核的收件箱里，会在**下一轮开头**被取走。
            # 不说清楚的话，用户会以为这句话石沉大海了。
            self._add_notice(f"有 {len(self._pending_injects)} 条插话没赶上这一轮，下一轮开头生效")
        self._set_input_mode()
        self._refresh_status()

    # ------------------------------------------------------------------
    # 流式文本
    # ------------------------------------------------------------------
    def _buffer_stream(self, text: str, *, reasoning: bool) -> None:
        if not text:
            return
        if reasoning:
            if self._reason_block is None:
                self._reason_block = self._transcript().add_folded(
                    render.reasoning_markup("", limit=self.fold_chars), render.KIND_REASONING)
            self._reason_buffer += text
        else:
            if self._stream_block is None:
                # 回答**按 Markdown 渲染**（代码块、列表、粗体都认）；
                # 流式续写就靠同一块的 `update`，实测每次约 0.23ms，20Hz 吃得下。
                self._stream_block = self._transcript().add_markdown("", render.KIND_ASSISTANT)
            self._stream_buffer += text

    def _flush_stream(self) -> None:
        """把缓冲的增量写进控件。**只在有东西可写时动控件**（这是每次刷新的成本所在）。"""
        if self._stream_block is not None and self._stream_buffer:
            self._stream_text += self._stream_buffer
            self._stream_buffer = ""
            # 回答是 Markdown **原文**，不转义 —— 转义会把 `**粗体**` 打成字面量。
            self._transcript().rewrite(self._stream_block, self._stream_text)
        if self._reason_block is not None and self._reason_buffer:
            self._reason_text += self._reason_buffer
            self._reason_buffer = ""
            # 思考块**边写边判要不要折**：超过阈值之后默认是折着的（只留开头一段），
            # 用户点开过就一直开着，不许替他折回去。
            self._transcript().rewrite_folded(
                self._reason_block,
                render.reasoning_markup(self._reason_text, limit=self.fold_chars))

    def _reset_stream(self) -> None:
        self._stream_block = None
        self._stream_text = ""
        self._stream_buffer = ""
        self._reason_block = None
        self._reason_text = ""
        self._reason_buffer = ""

    # ------------------------------------------------------------------
    # 状态栏与历史
    # ------------------------------------------------------------------
    def _tick(self) -> None:
        # 心跳只干一件事：把攒着的增量画上去（耗时不再进状态栏 ——「本轮结束」那行有）
        self._flush_stream()

    def _refresh_status(self) -> None:
        self.state.turns = self._turns_done + self._live_turn
        self.state.steps = self._steps
        self.state.usage = self._closed_usage + self._live_usage
        # 上下文占用：**借内核那个估算器**，不在这儿另写一份。
        # `Session.write_event` 每写一条事件都会把 `session.messages` 更新一次（`_apply_to_window`），
        # 所以它**就是**当前上下文窗口 —— 估算器认的那个"锚点"（最近一条带 usage 的 AI 消息）
        # 也在里面，于是"最近一次真实计数 ＋ 之后新增内容的粗估"这套规则一行都不用重写。
        # 不传 tools_schema：锚点那一支本来就不用它（固定开销已经含在那次真实计数里）；
        # 只有"还没发生过任何模型调用"时才差一点（那时它还是 0%，也就无所谓）。
        self.state.context_tokens = estimate_tokens(self.session.messages)
        self.state.context_max = self.harness.cfg.max_context_tokens
        self.query_one("#status", StatusBar).show(self.state)

    def _set_input_mode(self) -> None:
        self.query_one("#prompt", Input).placeholder = (
            BUSY_PLACEHOLDER if self.state.running else IDLE_PLACEHOLDER)

    def _replay(self, events: Sequence[Event]) -> None:
        """把一条老会话的时间线画回界面上，顺带把"历史累计"接到状态栏上。

        画什么由 :func:`tui.render.replay` 决定（**纯函数**，子会话视图与 ``--show-sub``
        走的是同一份），这里只把它算出来的块放上去、把它数出来的账记下来。
        """
        done = render.replay(events, fold_chars=self.fold_chars)
        self._transcript().draw(done.blocks)
        self._steps += done.steps                   # 回放也数：累计步数要接得上
        self._turns_done += done.turns
        if done.usage is not None:
            self._closed_usage = self._closed_usage + done.usage

    # ------------------------------------------------------------------
    def _transcript(self) -> Transcript:
        return self.query_one("#transcript", Transcript)

    def _add_notice(self, text: str) -> None:
        self._transcript().add(render.notice_block(text), render.KIND_NOTICE)

    def _add_error(self, event: Event) -> None:
        self._transcript().add(render.error_block(event), render.KIND_ERROR)


#: 模式 → 一句人话（切换时提示"换了之后意味着什么"）。
_MODE_HINT: dict[str, str] = {
    "readonly": "只读放行，写与执行一律拒绝",
    "ask": "写与执行都要问你",
    "accept_edits": "写文件直接放行，执行仍要问",
    "yolo": "全部放行，不再问你",
}
