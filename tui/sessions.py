# -*- coding: utf-8 -*-
"""会话树：把一个会话的**子子孙孙**读成一棵树，顺带算每个节点的状态与用量。

纯数据层（**不 import 界面框架**），所以不启动终端就能把它测干净 —— 和 :mod:`tui.render`
一个路子。控件那边只负责把这个结构放进一个 `Tree` 里。

## 一、为什么"看子会话"是读文件，而不是打开会话

`Harness.open(child_id)` 会把 ``harness._session`` **换成**子会话，之后 ``run_stream``
就往那条**委派记录**里写。子会话文件是"那次委派到底做了什么"的证据，被追加之后
``closed`` 也从真变假 —— 看一眼不该改东西。所以这条路只读：``SessionStore.events_of()``，
不碰 harness 的任何状态。

（顺带：``python -m tui --session <主会话>.sub-1`` 是**能跑通**的，因为子会话继承了父的
``meta.workspace``，工作区校验会过。那正是要避免的用法 —— 界面里的会话树才是正经入口。）

## 二、状态为什么是四态，而不是"进行中 / 已完成"

"文件末尾没有 ``session_end``"有**两种完全不同的原因**，混成一句话就是在说假话：

- 那次委派**还在跑**（父的这一轮还在飞）→ 正在进行；
- 那次委派**没写收尾就没了**（被打断、进程没了、工具超时）→ 未收尾。
  这是**唯一能看出"有一次委派没留下结论"的地方**，报成"进行中"会让它永远像在干活。

## 三、根节点为什么不显示状态

根就是**你现在所在的这条会话**，而且主会话的 ``session_end`` 要等 ``Harness.close()``
（退出程序）才写 —— 按文件判的话，你坐在里面的时候它会一直显示"未收尾"，那显然不对。
所以根只写"它是主会话"和它自己的账；它的状态在状态栏上。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.events import DataKey, EventType, Outcome
from src.core.message import Usage

from tui import render

# ---------------------------------------------------------------------------
# 状态
# ---------------------------------------------------------------------------
#: 那次委派还在跑（文件的末尾还没有 ``session_end``，而此刻确实有一轮在飞）。
RUNNING = "running"
#: 没有收尾就结束了（被打断 / 进程没了 / 超时）—— **不是**"还在跑"。
UNFINISHED = "unfinished"
#: 正常收尾。
DONE = "done"
#: 收尾了，但子树报的是失败（到轮次上限 / 出错 / 判为卡死）。
FAILED = "failed"

#: 状态 → ``(记号, 人话, 颜色)``。记号只用 **GBK 里有的**字形（``● ○ ◆ ▲``）：
#: 中文 Windows 的控制台默认是 cp936，画一个 GBK 里没有的字就是一次 ``UnicodeEncodeError``
#: （``•`` 与 ``▶`` 那两课见 `tui/console.py` 与 `tui/render.py`）。
#: 这一组还刻意和工具结果的记号同族：``◆`` 完成、``○`` 停在半路、``▲`` 出错。
STATE_WORD: dict[str, tuple[str, str, str]] = {
    RUNNING: ("●", "正在进行", "yellow"),
    UNFINISHED: ("○", "未收尾", "yellow"),
    DONE: ("◆", "完成", "green"),
    FAILED: ("▲", "失败", "red"),
}

#: 委派那段 prompt 在树里显示多少个字符。它是**区分同一个子代理的多次委派**的唯一线索
#: （``sub-1`` 和 ``sub-2`` 都是 explorer），所以必须露一点；但它是写给子代理看的完整任务
#: 描述（几百字），不能整段塞进一行。
PROMPT_CHARS = 36


# ---------------------------------------------------------------------------
# 从事件里读出来的事实
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Facts:
    """一条会话文件能告诉我们的事实。**纯函数算出来的**（见 :func:`facts_of`）。"""

    state: str | None = None
    turns: int = 0
    steps: int = 0
    usage: Usage | None = None
    duration_ms: int | None = None
    prompt: str = ""


def facts_of(events: Any, *, live: bool = False, main: bool = False) -> Facts:
    """一份事件序列 → :class:`Facts`。

    ``live`` ＝"此刻确实有一轮在跑"（界面那边只有它知道，所以从外面传进来）。
    ``main`` ＝ 这是**主控**会话的文件（不是谁的附属）。

    ★ 有两处判据是**照着 `src/harness/subagent.py` 写文件的方式**定的，不是猜的：

    - **子文件的第一对 ``tool_start`` / ``tool_result`` 是"委派括号"**：子代理一开跑，
      父那边就把"派给谁、问了什么"当作根事件写进去，收尾时再补一条结果 —— 它们是**父的**
      一次调用，不是子代理干的活。所以数步数时要减掉这一个（主控文件里没有这种东西）。
    - **最后一条 ``tool_result`` 的 ``usage`` 是那棵子树的全量**（子代理自己的调用 ＋
      压缩 ＋ 它再委派的下级），所以能用它就绝不再去累加下级的用量（加了就是重复计数）。
    """
    bracket = next((e for e in events if e.type == EventType.TOOL_START), None)
    result = next((e for e in reversed(events) if e.type == EventType.TOOL_RESULT), None)
    closed = bool(events) and events[-1].type == EventType.SESSION_END

    if main:
        # 主控会话的状态**不显示**（见模块开头的第三条）：它的 session_end 要等退出才写
        state: str | None = None
    elif closed:
        outcome = _outcome_of(result)
        state = DONE if outcome in ("", str(Outcome.OK)) else FAILED
    elif live:
        state = RUNNING
    else:
        state = UNFINISHED

    calls = sum(1 for e in events if e.type == EventType.TOOL_START)
    steps = calls - (1 if (bracket is not None and not main) else 0)

    usage = _usage_of(result) or _sum_usage(events, EventType.RUN_END) \
        or _sum_usage(events, EventType.ASSISTANT_MESSAGE)

    return Facts(
        state=state,
        # 轮：``run_end.turn`` 是那一次 run 用掉的轮数；还在跑时它还没落盘，
        # 那就看 ``model_start.turn``（它会一路涨到当前这一轮）。
        turns=max((e.turn or 0 for e in events
                   if e.type in (EventType.MODEL_START, EventType.RUN_END)), default=0),
        steps=max(0, steps),
        usage=usage,
        duration_ms=(result.data.get(DataKey.DURATION_MS) if result else None),
        prompt=("" if main or bracket is None
                else str((bracket.data.get("args") or {}).get("prompt", "") or "")),
    )


def _usage_of(event: Any) -> Usage | None:
    if event is None:
        return None
    got = event.data.get(DataKey.USAGE)
    return got if isinstance(got, Usage) else None


def _outcome_of(event: Any) -> str:
    """事件里的 ``outcome`` → 字符串。

    ★ **磁盘上读回来是普通字符串，内存里的活事件可能是 `Outcome` 枚举** —— 两种都得认。
    踩过：只写 ``getattr(raw, "value", "")`` 的话，从文件里读回来的 ``"failed"`` 会取不到
    ``value`` 而变成空串，于是**失败的委派被判成"完成"**（失败在这棵树里恰恰是最该一眼看见的）。
    """
    if event is None:
        return ""
    raw = event.data.get(DataKey.OUTCOME)
    return str(getattr(raw, "value", raw) or "")


def _sum_usage(events: Any, kind: str) -> Usage | None:
    total: Usage | None = None
    for event in events:
        if event.type != kind:
            continue
        got = (_usage_of(event) if kind == EventType.RUN_END
               else getattr(event.data.get(DataKey.MESSAGE), "usage", None))
        if isinstance(got, Usage):
            total = got if total is None else total + got
    return total


# ---------------------------------------------------------------------------
# 树
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SessionNode:
    """树里的一个会话。``children`` 是它的下一层（子代理派出去的下级）。"""

    id: str
    #: 显示名：根用完整 id，子节点用去掉根前缀的那截（``sub-1`` / ``sub-1.sub-1``）——
    #: 层级由树的缩进表达，id 里那串前缀在一行里只会占地方。
    name: str = ""
    agent: str = ""
    #: 四态之一；**主控会话是 ``None``**（见模块开头第三条）。
    state: str | None = None
    facts: Facts = field(default_factory=Facts)
    children: tuple["SessionNode", ...] = ()
    #: 这条会话**没有父**（是主控，不是谁的附属）。判据是文件头的 ``parent == -1``，
    #: 而不是"它在树里的位置"：`--show-sub <主会话 id>` 也要认得出它是个主控
    #: （那条路上没有树，只有一条会话）。
    main: bool = False

    @property
    def running(self) -> bool:
        return self.state == RUNNING


def build_tree(store: Any, root_id: str, *, live: bool = False) -> SessionNode:
    """以 ``root_id`` 为根，把它的**子子孙孙**读成一棵树。

    整个仓库的**文件头只扫一遍**（``iter_heads(parent=None)``，实测 0.4ms）就在内存里把
    父子关系接起来，而不是"每展开一层扫一次目录"。文件头里有 ``parent`` 与 ``meta.agent``，
    够搭树；**每个节点的事件各自读一遍**（实测 561KB 的主会话 6ms），因为状态与用量都在事件里。
    """
    heads = {head.session_id: head for head, _ in store.iter_heads(parent=None)}
    children: dict[str, list[str]] = {}
    for head in heads.values():
        if isinstance(head.parent, str):
            children.setdefault(head.parent, []).append(head.session_id)
    for ids in children.values():
        ids.sort()
    return _node(store, root_id, heads, children, live=live, root_id=root_id, seen=frozenset())


def read_one(store: Any, session_id: str, *, live: bool = False) -> SessionNode:
    """只读**一个**节点（不搭整棵树）：子会话视图刷新时用它 —— 别为了看一眼把整棵树再扫一遍。

    名字就用完整 id：那个"去掉根前缀"的短名（``sub-1``）只在树里有意义，进了查看界面之后
    人需要的是**完整 id**（要拿它去对文件、去 `--show-sub`）。
    """
    heads = {head.session_id: head for head, _ in store.iter_heads(parent=None)}
    return _node(store, session_id, heads, {}, live=live, root_id=session_id, seen=frozenset())


def _node(store: Any, session_id: str, heads: dict[str, Any], children: dict[str, list[str]],
          *, live: bool, root_id: str, seen: frozenset[str]) -> SessionNode:
    """一个节点：读它自己的事件，再把下一层挂上去。

    ``seen`` 防的是**环**：``head.parent`` 是文件里的一行数据，手工改过的文件完全可能
    指回自己或指成互指（``a.parent=b`` 且 ``b.parent=a``）。没有这道闸，搭树会直接爆栈
    —— 而"会话文件被手改过"不该让整个界面翻掉。
    """
    head = heads.get(session_id)
    is_main = bool(head is not None and head.parent == -1)
    try:
        events = store.events_of(session_id)
    except (OSError, ValueError):
        events = []                          # 读不了就当"没有收尾"：不假装它有结论
    facts = facts_of(events, live=live, main=is_main)

    kids: tuple[SessionNode, ...] = ()
    if session_id not in seen:
        kids = tuple(
            _node(store, child_id, heads, children, live=live, root_id=root_id,
                  seen=seen | {session_id})
            for child_id in children.get(session_id, ()))

    # 短名只对**树里的**子节点有意义：去掉根前缀（`sub-1` / `sub-1.sub-1`）。
    # 根（``session_id == root_id``）与单独读一条（``read_one``，那时 root_id 就是它自己）
    # 都用完整 id。
    name = session_id
    if session_id != root_id and session_id.startswith(f"{root_id}."):
        name = session_id[len(root_id) + 1:]

    return SessionNode(
        id=session_id,
        name=name,
        agent=str((head.meta.get("agent") if head else "") or ""),
        state=facts.state,
        facts=facts,
        children=kids,
        main=is_main,
    )


# ---------------------------------------------------------------------------
# 树上那一行怎么写
# ---------------------------------------------------------------------------
def node_label(node: SessionNode) -> str:
    """一个节点 → 树里那一行的标记串。

    子节点：``◆ sub-1 · explorer · 完成 · 2 轮 · 7 步 · 4.2k tok · “看看沙箱怎么做的”``
    根节点：``20260929-143211-80e606 · 主会话 · 3 轮 · 12 步 · 12.4k tok``

    **记号打头**（``● ○ ◆ ▲``）：一列扫下来就能看出"哪些还在跑、哪些停在半路"，
    不用逐个读字。名字紧跟其后（层级已经由缩进表达），状态词放在代理名之后 ——
    先认人、再看它怎么了。
    """
    mark, word, color = _state_of(node.state)
    head = f"{mark}[b]{render.escape(node.name)}[/b]" if mark \
        else f"[b]{render.escape(node.name)}[/b]"
    parts = [head]
    if node.main:
        parts.append("[cyan]主会话[/cyan]")
    elif node.agent:
        parts.append(f"[magenta]{render.escape(node.agent)}[/magenta]")
    if word:
        parts.append(f"[{color}]{word}[/{color}]")

    account = account_brief(node.facts)
    if account:
        parts.append(f"[dim]{account}[/dim]")
    if node.facts.prompt:
        parts.append(f"[dim]“{render.escape(render.brief(node.facts.prompt, PROMPT_CHARS))}”[/dim]")

    return "[dim] · [/dim]".join(parts)


def _state_of(state: str | None) -> tuple[str, str, str]:
    """状态 → ``(记号带尾空格, 人话, 颜色)``；根（``None``）给三个空串。"""
    if state is None:
        return "", "", ""
    mark, word, color = STATE_WORD.get(state, ("?", state, "white"))
    return f"[{color}]{mark}[/{color}] ", word, color


def account_brief(facts: Facts) -> str:
    """``2 轮 · 7 步 · 4.2k tok``（有什么写什么：还在跑的委派可能一样都还没有）。"""
    parts: list[str] = []
    if facts.turns:
        parts.append(f"{facts.turns} 轮")
    if facts.steps:
        parts.append(f"{facts.steps} 步")
    if facts.usage is not None:
        parts.append(f"{render.fmt_tokens(facts.usage.total_tokens)} tok")
    return " · ".join(parts)


def header_line(node: SessionNode) -> str:
    """只读查看界面抬头**第一行**：它是谁 ＋ 它怎么了 ＋ 花了多少。

    比树上那一行详细：把完整 id 和耗时也写出来 —— 进到这一层的人已经在看细节了。
    """
    mark, word, color = _state_of(node.state)
    parts = [f"{mark}[b]{render.escape(node.id)}[/b]" if mark
             else f"[b]{render.escape(node.id)}[/b]"]
    if word:
        parts.append(f"[{color}]{word}[/{color}]")
    if node.main:
        parts.append("[cyan]主会话[/cyan]")
    elif node.agent:
        parts.append(f"[magenta]{render.escape(node.agent)}[/magenta]")
    account = account_brief(node.facts)
    if account:
        parts.append(f"[dim]{account}[/dim]")
    if node.facts.duration_ms is not None:
        parts.append(f"[dim]{render.fmt_duration(int(node.facts.duration_ms))}[/dim]")
    return "[dim] · [/dim]".join(parts)


def header_prompt(node: SessionNode, limit: int = 160) -> str:
    """抬头**第二行**：这次委派到底让它干什么。

    ★ 这一段必须在这儿露面：时间线里那个"委派括号"（根 ``tool_start``）被 :func:`strip_bracket`
    去掉了（见那里的说明），而那句 prompt 只存在它身上 —— 不在这儿补回来，看子会话的人
    就永远不知道它是被派去干什么的。
    """
    if not node.facts.prompt:
        return ""
    text = render.escape(render.brief(node.facts.prompt, limit))
    return f"[dim]委派内容：[/dim]“{text}”"


def strip_bracket(events: list[Any]) -> list[Any]:
    """去掉子文件里那对**委派括号**（第一个 ``tool_start`` 与关掉它的那条 ``tool_result``）。

    它们是**父的那一次调用**（"派给谁、问了什么"），不是子代理干的活：留在子会话的时间线上，
    会让人以为"这个子代理又派了一次"。抬头那两行已经把这层意思说全了
    （:func:`header_line` ＋ :func:`header_prompt`），所以这里去掉。

    只认**文件里第一个** ``tool_start``，而且它得是 ``kind=subagent``（``src/harness/subagent.py``
    就是这么写的）—— 子代理自己再派出去的 ``task`` 出现在更后面，不会被误伤。
    """
    bracket = next((e for e in events if e.type == EventType.TOOL_START), None)
    if bracket is None or str(bracket.data.get(DataKey.TOOL_TYPE, "")) != "subagent":
        return list(events)
    span = str(bracket.span)
    out: list[Any] = []
    dropped = False
    for event in events:
        if event is bracket:
            continue
        if (not dropped and event.type == EventType.TOOL_RESULT
                and str(event.span) == span):
            dropped = True                    # 关掉括号的那条结果
            continue
        out.append(event)
    return out


def walk(node: SessionNode) -> list[SessionNode]:
    """整棵树摊平成清单（自己在前、子节点在后）。给"按 id 找节点"这类用法兜底。"""
    out = [node]
    for child in node.children:
        out.extend(walk(child))
    return out


def find(node: SessionNode, session_id: str) -> SessionNode | None:
    """按 id 找节点（就地往下找，找不到就是 ``None``）。"""
    return next((item for item in walk(node) if item.id == session_id), None)
