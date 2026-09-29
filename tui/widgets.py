# -*- coding: utf-8 -*-
"""界面控件：时间线、状态栏、审批弹窗、会话树与子会话视图。

这几个都刻意做得很薄 —— "这句话该怎么写"在 :mod:`tui.render` 里，"这棵树长什么样"在
:mod:`tui.sessions` 里（两份都是纯数据、可单测），这里只负责把字符串放进控件、
把按键变成动作。
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Markdown, Static, Tree

from src.core.events import ApprovalDecision, now_ms

from tui import render, sessions
from tui.render import StatusState, approval_facts, approval_head, status_line


class Transcript(VerticalScroll):
    """时间线：内容一块一个 :class:`Static`，正在流式写入的那块**就地更新**。

    为什么不每次增量都新建一个块：一条回答会有几百条 ``text_delta``，一块一条等于把
    时间线炸成几百个控件，滚动和布局都会跟着变慢。**一块装一条回答**，只有"换说话人"
    或者"换动作"时才新起一块。
    """

    def add(self, markup: str, kind: str) -> Static:
        """新起一块。``kind`` 同时是 CSS 类名（见 ``tui/app.py`` 的 ``CSS``）。"""
        block = Static(markup, classes=f"block {kind}")
        self.mount(block)
        self._follow(block)
        return block

    def add_markdown(self, text: str, kind: str) -> Markdown:
        """新起一块**按 Markdown 渲染**的内容（助手的回答）。

        流式续写也走它：`Markdown.update()` 实测约 0.23ms 一次（200 次追加 46ms），
        20Hz 的刷新频率完全吃得下 —— 所以不必"先纯文本、完成后再转 Markdown"那套。
        """
        block = Markdown(text, classes=f"block {kind}")
        self.mount(block)
        self._follow(block)
        return block

    def add_tool(self, view: render.ToolView, kind: str) -> ToolBlock:
        """新起一块工具调用（头 ＋ 折叠正文）。"""
        block = ToolBlock(view, classes=f"block {kind}")
        self.mount(block)
        self._follow(block)
        return block

    def add_folded(self, folded: Any, kind: str) -> Foldable:
        """新起一块**可折叠**的标记文本（思考过程用它；工具调用用 :meth:`add_tool`）。"""
        block = Foldable(folded, classes=f"block {kind}")
        self.mount(block)
        self._follow(block)
        return block

    def draw(self, blocks: Sequence[render.Block]) -> None:
        """一次把一批块画上去（**回放与子会话视图共用这一条路**）。

        按块的性质挑控件（四选一，见 :class:`tui.render.Block`）。实时那条路不走这里 ——
        它得一块一块地边收边改（流式续写、结果回来时就地重画），那是另一套动作。
        """
        for block in blocks:
            if block.tool is not None:
                self.add_tool(block.tool, block.kind)
            elif block.folded is not None:
                self.add_folded(block.folded, block.kind)
            elif block.markdown:
                self.add_markdown(block.text, block.kind)
            else:
                self.add(block.text, block.kind)

    def rewrite(self, block: Any, content: str) -> None:
        """就地改一块的内容（流式续写）：`Static` 收标记、`Markdown` 收 Markdown 原文。"""
        block.update(content)
        self._follow(block)

    def rewrite_folded(self, block: Foldable, folded: Any) -> None:
        """就地改一块折叠内容（**保持用户选的那个状态**，见 `Foldable.set_folded`）。"""
        block.set_folded(folded)
        self._follow(block)

    def _follow(self, block: Static) -> None:
        """跟着滚到底 —— **但只在用户本来就在底部时**。

        无条件滚动会让"往上翻着看历史"变成不可能：模型每吐一个字就把视图拽回底部。
        用"滚动位置离底部还有多远"来判，误差留三行，够容下正在长高的那一块。
        """
        at_bottom = self.scroll_offset.y >= self.max_scroll_y - 3
        if at_bottom or self.max_scroll_y <= 0:
            self.scroll_end(animate=False)


class StatusBar(Static):
    """一行状态：模式 · 模型 · 会话 · 轮次 · token（含缓存命中率）· 本轮耗时 · 在干什么。"""

    def show(self, state: StatusState) -> None:
        self.update(status_line(state))


class ToolBlock(Vertical):
    """一次工具调用：头两行 ＋ **可折叠**的正文，正文交给 Markdown 渲染。

    折叠态只给一行"（N 行 · M 字符 · 点开看）"，展开态才是正文 —— 工具结果常是几十行文件内容
    或日志，摊在时间线上会把对话淹掉。

    与 :class:`Foldable` 的分工：那个是**同一段标记文本的两份写法**（思考块用它，一份折叠一份展开）；
    这个是"头（标记）＋ 正文（Markdown）"，因为正文要交给 Markdown 解析器，不能当标记串塞进
    `Static`。两者共用同一套交互约定：整块可点、聚焦后 `Enter`/`Space` 也能切、`[+]` / `[-]` 记号。
    """

    can_focus = True

    BINDINGS = [
        Binding("enter", "toggle", "展开/收起", show=False),
        Binding("space", "toggle", "展开/收起", show=False),
    ]

    def __init__(self, view: render.ToolView, *, classes: str = "") -> None:
        super().__init__(classes=classes)
        self._view = view
        #: 用户手动选过吗。``None`` = 还没选过 —— 工具正文**默认折着**（用户定的）。
        self._chosen: bool | None = None

    def compose(self) -> ComposeResult:
        yield Static(self._view.head, classes="tool-head")
        # 提示与正文**一开始就都建好**（哪怕是空的）：结果回来时只改内容、不挂载新控件，
        # 块的身份在流式途中就不会变（上一次的坑：换了存储结构却忘了块引用）。
        yield Static(self._hint_markup(), classes="tool-hint")
        yield Markdown(self._view.body, classes="tool-body")

    def on_mount(self) -> None:
        self.set_class(self.foldable, "foldable")
        self._apply()

    @property
    def opened(self) -> bool:
        """正文露着吗。**没手动选过就是折着**（工具正文一律默认折叠）。"""
        return bool(self._chosen)

    @property
    def foldable(self) -> bool:
        """有没有正文可折（被拒绝、空结果就没有，此时也不显示记号、不响应点击）。"""
        return bool(self._view.foldable)

    @property
    def body(self) -> str:
        """正文原文（测试与排障用；渲染交给 Markdown）。"""
        return self._view.body

    def update_view(self, view: render.ToolView) -> None:
        """结果回来了：换掉头、提示与正文，**保持用户选的那个状态**。"""
        self._view = view
        self.query_one(".tool-head", Static).update(view.head)
        self.query_one(".tool-hint", Static).update(self._hint_markup())
        self.query_one(".tool-body", Markdown).update(view.body)
        self.set_class(self.foldable, "foldable")
        self._apply()

    def _hint_markup(self) -> str:
        return self._view.hint_open if self.opened else self._view.hint_closed

    def _apply(self) -> None:
        """按当前状态决定"显示提示还是显示正文"。

        提示行**两种状态都显示**（折着时是 `[+] （… 点开看）`，展开时是 `[-]`）：
        它是"能点"的入口，展开之后把它藏掉，人就得先去猜怎么收回去。
        正文只在展开时显示 —— **折着时一点都不露**。
        """
        self.query_one(".tool-hint", Static).display = self.foldable
        self.query_one(".tool-body", Markdown).display = self.foldable and self.opened

    def action_toggle(self) -> None:
        if not self.foldable:
            return
        self._chosen = not self.opened
        self.query_one(".tool-hint", Static).update(self._hint_markup())
        self._apply()

    def on_click(self, event: Any) -> None:
        if self.foldable:
            event.stop()
            self.focus()
            self.action_toggle()


class Foldable(Static):
    """一段**可以折起来**的内容：默认折着（只给开头一段），点一下展开全文，再点一下收回。

    三条取舍：

    - **"写什么"不在这里**：折叠态与展开态各是一份现成的标记，由 `tui/render.py` 按内容算
      （包括"到底值不值得折"）。这里只管在两者之间切。
    - **整块可点，不是只点那个记号**：一个字符宽的目标在终端里很难点中，而这一块本来就只有
      "展开/收起"一个动作 —— 点哪儿都一样。
    - **焦点 ＋ ``Enter``/``Space`` 也能切**：不是每个终端都开了鼠标上报，键盘得留一条路。
      ``foldable=False``（内容没超限）时**不显示记号、也不响应** —— 给一个点了没反应的记号，
      比不给记号更糟。
    """

    can_focus = True

    BINDINGS = [
        Binding("enter", "toggle", "展开/收起", show=False),
        Binding("space", "toggle", "展开/收起", show=False),
    ]

    def __init__(self, folded: Any, *, classes: str = "") -> None:
        self._folded = folded
        #: 用户手动选过吗。``None`` = 还没选过，那就按内容决定（**超限就折着**）；
        #: 一旦他点过，那就是"人的意思"—— 后续内容再变也不许替他改回去。
        self._chosen: bool | None = None
        super().__init__(self._markup(), classes=f"{classes} foldable" if folded.foldable else classes)

    def _markup(self) -> str:
        return self._folded.expanded if self.opened else self._folded.collapsed

    @property
    def opened(self) -> bool:
        """现在展开着吗。**没手动选过时按内容定**：没超限就展开（没什么可折），超了就折着。"""
        if self._chosen is None:
            return not self._folded.foldable
        return self._chosen

    @property
    def foldable(self) -> bool:
        """这块内容值不值得折（内容没超限就是 ``False``，此时它不显示记号、也不响应点击）。"""
        return bool(self._folded.foldable)

    def set_folded(self, folded: Any) -> None:
        """内容变了（流式续写）就重画，但**保持用户选的那个状态**。"""
        self._folded = folded
        self.set_class(folded.foldable, "foldable")
        self.update(self._markup())

    def action_toggle(self) -> None:
        if not self._folded.foldable:
            return
        self._chosen = not self.opened
        self.update(self._markup())

    def on_click(self, event: Any) -> None:
        if self._folded.foldable:
            event.stop()
            self.focus()
            self.action_toggle()


@dataclass(frozen=True)
class ApprovalAnswer:
    """弹窗的答案：**这是第几条请求** ＋ 决定 ＋ 谁做的。

    带上 ``index`` 是为了让 :class:`~tui.bridge.RunBridge` 能丢掉过期的决定 ——
    超时已经按默认值往下走了，界面才把按钮的答案送回来，那个答案必须作废。
    """

    index: int
    decision: ApprovalDecision
    by: str = "human"


class ApprovalModal(ModalScreen[ApprovalAnswer]):
    """审批弹窗：**要批什么、为什么问、不回答会怎样、还有几秒**。

    按键：``y`` / ``n`` / ``Esc``（Esc 等同拒绝 —— 弹窗上的"跑掉"必须落在安全那一侧）。
    **不倒计时自己关**：到点由 :class:`~tui.bridge.RunBridge` 按默认值继续，这里只画
    剩余时间。两处各算一次超时，就会出现"界面还开着、run 已经走了"的错位。
    """

    BINDINGS = [
        Binding("y", "approve", "批准"),
        Binding("n", "deny", "拒绝"),
        Binding("escape", "deny", "拒绝"),
    ]

    def __init__(self, request: Any, index: int) -> None:
        super().__init__()
        self.request = request
        self.index = index

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-box"):
            yield Static(approval_head(self.request), id="approval-head")
            with Vertical(id="approval-body"):
                for line in approval_facts(self.request):
                    yield Static(line, classes="approval-fact")
                yield Static("", id="approval-count")
            with Horizontal(id="approval-buttons"):
                yield Button("批准 (y)", variant="success", id="approve")
                yield Button("拒绝 (n)", variant="error", id="deny")

    def on_mount(self) -> None:
        self._tick()
        self.set_interval(0.25, self._tick)
        # 焦点给"拒绝"：焦点在哪个按钮上，回车就等于按了它 —— 默认该落在安全那一侧。
        self.query_one("#deny", Button).focus()

    def _tick(self) -> None:
        left_ms = self.request.expires_ts - now_ms()
        label = self.query_one("#approval-count", Static)
        if left_ms <= 0:
            consequence = "放行" if self.request.proposed else "拒绝"
            label.update(f"[dim]已超时 —— 按默认值[b]{consequence}[/b]继续[/dim]")
            return
        seconds = max(1, int(left_ms / 1000))
        label.update(f"剩余 {seconds} 秒")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "approve":
            self.action_approve()
        else:
            self.action_deny()

    def action_approve(self) -> None:
        self.dismiss(ApprovalAnswer(self.index, ApprovalDecision.APPROVED))

    def action_deny(self) -> None:
        self.dismiss(ApprovalAnswer(self.index, ApprovalDecision.DENIED))


# ---------------------------------------------------------------------------
# 会话树：委派出去的子子孙孙
# ---------------------------------------------------------------------------
class SessionTree(Tree):
    """会话树控件。**只换掉字形**，别的都用 textual 现成的。

    ★ textual 默认的收起标记是 ``▶``（U+25B6）—— **不在 GBK 里**，中文 Windows 的控制台上
    画它就是一次 ``UnicodeEncodeError``（``•`` 与 ``▶`` 那两课见 `tui/console.py`）。
    换成 ``[+]`` / ``[-]`` 顺带和时间线上折叠块**长成同一个记号**：两处都是"这里能展开"。
    引导线那些 ``├ └ │ ─`` 本来就在 GBK 里，不用动（已逐字验过）。
    """

    ICON_NODE = "[+] "
    ICON_NODE_EXPANDED = "[-] "

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # `auto_expand` **不是构造参数**（`Tree.__init__` 的签名里没有它），只能造完再设。
        # 关掉它是刻意的：开着的话 `Enter` 在一个"自己也派过子代理"的会话上会**边展开边打开**，
        # 两件事撞在一起 —— 展开交给 `Space`（`Tree` 自带的键位）。
        self.auto_expand = False


class SessionTreeModal(ModalScreen[None]):
    """会话树：以**当前打开的会话**为根，把子代理的子子孙孙摊开。

    键：``↑↓`` 走 · ``Space`` 展开/收起 · ``Enter``（或点节点那一行）打开选中的会话 ·
    ``Esc`` 关掉。``Enter`` 只负责"打开"，展开是 ``Space``（理由见 :class:`SessionTree`）。

    打开的是 :class:`SubSessionModal`（**只读**），一次一个；看完 ``Esc`` 回到这棵树，
    再选别的。层级因此不会迷路：你永远知道自己在哪一层。

    **运行中会自己刷新**（每秒一次）：委派还在跑的时候，状态要从"正在进行"变成"已完成"，
    不该让人退出去再进来。
    """

    BINDINGS = [Binding("escape", "close", "关闭")]

    def __init__(self, store: Any, root_id: str, *, fold_chars: int,
                 live: Callable[[], bool], refresh_s: float = 1.0) -> None:
        super().__init__()
        self.store = store
        self.root_id = root_id
        self.fold_chars = fold_chars
        self.live = live
        self.refresh_s = refresh_s
        self.session_tree = sessions.build_tree(store, root_id, live=live())
        #: 哪些 id 是展开着的（**跨刷新保留**：重建这棵树不该把用户展开的状态弄丢）。
        self._open: set[str] = {root_id}
        self._cursor: str = root_id
        self._signature: Any = None
        #: 上次刷新时**还有节点在跑**吗。见 :meth:`_refresh`：它决定"这棵树还要不要盯下去"。
        self._saw_running: bool = any(node.running for node in sessions.walk(self.session_tree))

    def compose(self) -> ComposeResult:
        with Vertical(id="tree-box"):
            yield Static("", id="tree-head")
            yield SessionTree(self.root_id, id="tree")
            yield Static("", id="tree-foot")

    def on_mount(self) -> None:
        self._fill()
        self.query_one("#tree", SessionTree).focus()
        if self.refresh_s > 0:
            self.set_interval(self.refresh_s, self._refresh)

    # ---- 画面 ----
    def _fill(self) -> None:
        """把整棵树画上去（重建时保持"展开了哪些、光标在哪"）。"""
        widget = self.query_one("#tree", SessionTree)
        self._signature = _signature(self.session_tree)
        widget.clear()
        widget.root.label = sessions.node_label(self.session_tree)
        widget.root.data = self.session_tree
        widget.root.expand()
        _add_children(widget.root, self.session_tree, open_ids=self._open)
        self._update_head()
        target = _find_widget_node(widget.root, self._cursor)
        if target is not None:
            widget.move_cursor(target)

    def _update_head(self) -> None:
        total = len(sessions.walk(self.session_tree)) - 1
        running = sum(1 for n in sessions.walk(self.session_tree) if n.running)
        head = (f"会话树 · 根 {self.root_id} · 子会话 {total} 个"
                + (f"（{running} 个正在进行）" if running else ""))
        self.query_one("#tree-head", Static).update(f"[b]{render.escape(head)}[/b]")
        self.query_one("#tree-foot", Static).update(
            "[dim]↑↓ 选 · Space 展开/收起 · Enter（或点一下）查看选中的子会话 · "
            "Enter 在根上＝回到时间线 · Esc 关闭[/dim]")

    # ---- 刷新 ----
    def _refresh(self) -> None:
        """重读这棵树，**只在"还可能变"的时候读**。

        停止的条件不是"现在没有 run 在跑"，而是"**现在没有 run 在跑，而且上次也没看见谁在跑**"。
        ★ 只判前者会漏掉一个很常见的场景：委派跑到一半时打开树（那时显示"正在进行"），
        然后它跑完了 —— `live()` 变成假，刷新跟着停下，**树上就永远停在"正在进行"**，
        非退出去再进来一次不可。文件只有 run 在跑的时候才会变，所以"上次看见在跑"之后
        多读一次，读到尘埃落定就自然停。

        重读之后先比一份"签名"（有哪些节点、各自什么状态、几个轮几步）：一样就不重建 ——
        否则每秒重建一次会把用户展开的状态和滚动位置反复抖掉。
        """
        if not self.live() and not self._saw_running:
            return
        fresh = sessions.build_tree(self.store, self.root_id, live=self.live())
        self._saw_running = any(node.running for node in sessions.walk(fresh))
        if _signature(fresh) == self._signature:
            return
        self.session_tree = fresh
        self._fill()

    # ---- 交互 ----
    def on_tree_node_expanded(self, event: Tree.NodeExpanded) -> None:
        node = getattr(event.node, "data", None)
        if isinstance(node, sessions.SessionNode):
            self._open.add(node.id)

    def on_tree_node_collapsed(self, event: Tree.NodeCollapsed) -> None:
        node = getattr(event.node, "data", None)
        if isinstance(node, sessions.SessionNode) and node.id != self.root_id:
            self._open.discard(node.id)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """``Enter`` 与"点节点那一行"都会走到这里（方向键只发 ``NodeHighlighted``）。

        根节点＝"你正在这条会话里"，选它等于**回到时间线**（树是自己按一下 ``Ctrl+T``
        开出来的，收回去的动作就该在这儿）。
        """
        node = getattr(event.node, "data", None)
        if not isinstance(node, sessions.SessionNode):
            return
        self._cursor = node.id
        if node.id == self.root_id:
            # 选到**根**（你正在这条会话里）＝ 回到时间线：树是自己按一下 Ctrl+T 开出来的，
            # 收回去的动作就该在这儿。
            self.dismiss()
            return
        self.app.push_screen(SubSessionModal(
            self.store, node.id, root_id=self.root_id,
            fold_chars=self.fold_chars, live=self.live, refresh_s=self.refresh_s))

    def action_close(self) -> None:
        self.dismiss()


class SubSessionModal(ModalScreen[None]):
    """**只读**地看一条子会话：它就是那条 `.sub-N.jsonl` 的时间线。

    为什么只读（而不是"接着它聊"）：子代理内部**不审批**（给它判定链，但不给审批入口），
    所以"继续"要么跑不动、要么行为和在主会话里的预期不一致；而且真要接着做，正确的事是
    **新派一次**，不是改这条委派记录。

    抬头一行写"它是谁 ＋ 它怎么了 ＋ 花了多少"，下面就是时间线 —— 和主时间线同一套渲染
    （:func:`tui.render.replay`），所以工具块、折叠、Markdown 全都一样。
    """

    BINDINGS = [
        Binding("escape", "close", "返回会话树"),
        Binding("r", "reload", "刷新"),
    ]

    def __init__(self, store: Any, session_id: str, *, root_id: str, fold_chars: int,
                 live: Callable[[], bool], refresh_s: float = 1.0) -> None:
        super().__init__()
        self.store = store
        self.session_id = session_id
        self.root_id = root_id
        self.fold_chars = fold_chars
        self.live = live
        self.refresh_s = refresh_s
        self._seen: Any = None
        #: 上一次读到的是"正在进行"吗（见 :meth:`_refresh`：跑完那一刻要再读一次才收得住）。
        self._saw_running = False

    def compose(self) -> ComposeResult:
        with Vertical(id="view-box"):
            yield Static("", id="view-head")
            yield Transcript(id="view-transcript")
            yield Static("", id="view-foot")

    async def on_mount(self) -> None:
        await self.action_reload()
        if self.refresh_s > 0:
            self.set_interval(self.refresh_s, self._refresh)

    def _load(self) -> tuple[sessions.SessionNode, render.Replay]:
        node = sessions.read_one(self.store, self.session_id, live=self.live())
        try:
            events = self.store.events_of(self.session_id)
        except OSError:
            events = []
        # 去掉那对"委派括号"：它是**父**的那次调用，抬头两行已经把它说全了
        # （留在时间线上会让人以为"这个子代理又派了一次"）。主控会话那条路上没有括号，
        # `strip_bracket` 也认得出（它只认 kind=subagent 的第一个 tool_start，而主控的
        # 第一次调用几乎不可能是它）—— 这里仍然按 main 明确分开，避免主控真的先派一次时被误删。
        events = events if node.main else sessions.strip_bracket(events)
        return node, render.replay(events, fold_chars=self.fold_chars)

    async def action_reload(self) -> None:
        """重画一遍（刷新与手动 ``r`` 都走这里）。

        ``remove_children()`` 要 **await**：它是"排队等着拆掉"的异步动作，不等它就去挂新块，
        两边会插在一起（屏幕上出现重影）。所以这个动作是 async 的 ——
        textual 对 async 的 action 会自己 await（键位与定时器都吃）。
        """
        node, done = self._load()
        self._seen = (len(done.blocks), node.state, done.steps)
        self._saw_running = node.running
        head = sessions.header_line(node)
        prompt = sessions.header_prompt(node)
        self.query_one("#view-head", Static).update(f"{head}\n{prompt}" if prompt else head)
        self.query_one("#view-foot", Static).update(
            "[dim]只读 · 这是那次委派留下的记录，不能在这里接着聊 · "
            f"Esc 返回会话树 · r 刷新{' · 运行中会自动刷新' if self.live() else ''}[/dim]")
        view = self.query_one("#view-transcript", Transcript)
        await view.remove_children()
        view.draw(done.blocks)
        if not done.blocks:
            view.add(render.notice_block("（这条会话还没有内容）"), render.KIND_NOTICE)

    async def _refresh(self) -> None:
        """重读这一条，**只在"还可能变"的时候读**（判据同会话树的 :meth:`SessionTreeModal._refresh`：
        "没在跑"**且**"上次也没看见在跑"才停 —— 否则跑完那一刻会停在"正在进行"不动）。

        而且**只在真的变了的时候**重画：不然用户展开的折叠块会被每秒抖回去。
        """
        if not self.live() and not self._saw_running:
            return
        node = sessions.read_one(self.store, self.session_id, live=self.live())
        try:
            count = len(self.store.events_of(self.session_id))
        except OSError:
            return
        if self._seen is not None and (count, node.state) == (self._seen[0], self._seen[1]):
            return
        await self.action_reload()

    def action_close(self) -> None:
        self.dismiss()


# ---------------------------------------------------------------------------
# 会话树那几个小工具
# ---------------------------------------------------------------------------
def _add_children(parent: Any, node: sessions.SessionNode, *, open_ids: set[str]) -> None:
    """递归把子节点挂上（``open_ids`` 里的保持展开 —— 跨刷新要用）。"""
    for child in node.children:
        kid = parent.add(sessions.node_label(child), data=child)
        if child.id in open_ids:
            kid.expand()
        if child.children:
            _add_children(kid, child, open_ids=open_ids)


def _find_widget_node(widget_root: Any, session_id: str) -> Any:
    """在控件的节点树里按 id 找回那个节点（刷新之后要把光标放回原处）。"""
    stack = [widget_root]
    while stack:
        item = stack.pop()
        data = getattr(item, "data", None)
        if isinstance(data, sessions.SessionNode) and data.id == session_id:
            return item
        stack.extend(item.children)
    return None


def _signature(node: sessions.SessionNode) -> tuple:
    """树的"样子"的指纹：有哪些节点、各自什么状态、几个轮几步。

    刷新时拿它判断"要不要重建" —— 没有它就得每秒重建一次，用户展开的状态和滚动位置
    会被反复抖掉。
    """
    return tuple((item.id, item.state, item.facts.turns, item.facts.steps)
                 for item in sessions.walk(node))

