# -*- coding: utf-8 -*-
"""事件 → 界面上要显示的那些行。

**纯函数，不 import 界面框架**：所有"这句话该怎么写"的判断都留在这里，于是不启动终端
也能把它们测干净（`tests/test_tui/test_render.py`）。:mod:`tui.widgets` 只负责把这些
字符串放进控件里。

## 颜色与字符

- 颜色用 textual 的行内标记（``[dim]`` / ``[red]`` 一类），由 `Static` 渲染。
  **所有来自事件的内容都必须先 :func:`escape`** —— 模型吐一句 ``[b]``，不转义就会被
  当成标记解析掉（见 :func:`escape` 的说明）。
- 字形只用 **GBK 里有的那些**（``·`` ``│`` ``┌`` ``└`` ``◆`` ``▲`` ``○``）：界面在中文
  Windows 上默认跑在 cp936 下，画一个 GBK 里没有的字符就是一次 ``UnicodeEncodeError``
  （``•`` 那一课见 `tui/console.py` 的说明）。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from src.core.events import DataKey, EventType, Outcome, StopReason, now_ms
from src.core.message import Usage

_MARKUP_OPEN = re.compile(r"\[")


def escape(text: str) -> str:
    """把一段**内容**里的 ``[`` 全部转义，免得它被当成标记解析掉。

    ★ **不能用 `rich.markup.escape` 代替**：它只转义"看起来像合法标记"的那些
    （``[b]`` / ``[/b]`` / ``[#fff]``），``[ERROR]`` ``[WARN]`` 这类**大写开头的方括号
    它一个都不管**。而 textual 的标记解析器对 ``[xxx]`` 是**照单全收**的：认不出来的
    标签既不报错也不显示，直接**当标记吃掉** —— 模型说的一句 ``[ERROR] 编译失败``
    到了屏幕上就只剩 `` 编译失败``，命令输出里的 ``[INFO]`` 也一样。

    少几个字比报错难发现得多（报错至少有人修），所以这里一律转义。
    """
    return _MARKUP_OPEN.sub(r"\\[", text)

#: 时间线上一个块的种类。**与 CSS 类名一一对应**（见 `tui/app.py` 的 ``CSS``）。
KIND_USER = "user"
KIND_ASSISTANT = "assistant"
KIND_REASONING = "reasoning"
KIND_TOOL = "tool"
KIND_NOTICE = "notice"
KIND_ERROR = "error"

#: 工具结果的"结束方式" → 记号 ＋ 人话。取值见 :class:`~src.core.events.Outcome`。
#: **被拒 / 被拦不算失败**："工具没跑"和"跑了但失败"是两件事，颜色也不该一样。
OUTCOME_MARK: dict[str, tuple[str, str]] = {
    Outcome.OK: ("◆", "完成"),
    Outcome.FAILED: ("▲", "失败"),
    Outcome.DENIED: ("○", "被权限拒绝"),
    Outcome.BLOCKED: ("○", "被钩子拦下"),
    Outcome.TIMEOUT: ("▲", "超时"),
    Outcome.CANCELLED: ("○", "取消"),
    Outcome.INTERRUPTED: ("○", "打断"),
}

#: run 为什么停下。取值见 :class:`~src.core.events.StopReason`。
STOP_WORD: dict[str, str] = {
    StopReason.END_TURN: "正常收尾",
    StopReason.MAX_TURNS: "到轮次上限",
    StopReason.INTERRUPTED: "被打断",
    StopReason.BUDGET_EXCEEDED: "上下文压不动了",
    StopReason.STALLED: "判为卡死",
    StopReason.ERROR: "出错",
    StopReason.CANCELLED: "取消",
}

#: 风险等级 → 人话。取值见 `src/harness/permissions.py`。
RISK_WORD: dict[str, str] = {"read": "只读", "write": "写", "exec": "执行", "spawn": "派生"}

#: 工具 → ``(显示名, 要显示的参数)``。**这张表是"界面怎么念一次调用"的唯一出处。**
#:
#: 工具面里**本地**那些是固定的（编码工具 ＋ 沙箱 bash ＋ 技能 ＋ 子代理），所以按名字查表
#: 比"猜哪个参数最像主语"稳：``write_file`` 的两个参数里永远该显示 ``path``（``content``
#: 可能有几十 KB），``grep`` 该同时显示 ``pattern`` 与 ``path``。
#:
#: 它**刻意不复用**权限层那份 ``invocation_summary``：那份是给**记账**用的可匹配字符串
#: （一次调用一行，以后要按它比对），这里是给人看的显示名 —— 两者的取舍不同，硬凑成一份
#: 会两头都别扭。表里查不到的两个形状（MCP 远端工具、`extra_tools` 塞进来的自定义工具）
#: 走 :func:`tool_label` 的后两档。
TOOL_LABELS: dict[str, tuple[str, tuple[str, ...]]] = {
    "read_file": ("read", ("path",)),
    "write_file": ("write", ("path",)),
    "edit_file": ("edit", ("path",)),
    "list_dir": ("list", ("path",)),
    "glob_files": ("glob", ("pattern",)),
    "grep": ("grep", ("pattern", "path")),
    "todo_write": ("todo", ("content",)),
    "read_skill": ("read skill", ("name",)),
    "sandbox_bash": ("sandbox bash", ("command", "timeout_ms")),
    # 委派那一行只显示"派给谁"：`prompt` 是一整段自包含的任务描述（几百字），
    # 塞进标题行就看不见别的了 —— 它在该子代理自己的会话文件里。
    "task": ("task", ("agent",)),
}

#: 一个参数值在标题行里最多显示多少个字符（`todo_write` 的 content 可以很长）。
LABEL_VALUE_CHARS = 80

#: MCP 远端工具名的前缀。**它是个约定，不是猜的**：`src/harness/mcp.py` 的 `_mcp_tool_name`
#: 按 ``f"mcp__{server}__{tool}".lower()`` 造名字（再把非 ``[a-z0-9_]`` 的字符换成 ``_``），
#: 所以光看名字就知道这个工具是从哪个 server 来的 —— 界面把这层出处念出来（见 :func:`_mcp_label`）。
MCP_PREFIX = "mcp__"

#: 思考内容**超过多少个字符就折起来**（折叠态只显示前这么多字符 ＋ ``…``）。
#: 400 是"一眼能扫完"的量级：短于此的不折，也就不给记号（见 :func:`reasoning_markup`）。
#: **工具结果不走这个阈值** —— 它的正文**一律**折起来（见 :func:`tool_result_markup`）。
FOLD_CHARS = 400

#: 折叠记号：**关**（可以点开）与**开**（可以收回）。放在内容行首，点哪儿都能切。
#:
#: ★ 用 ASCII 的 ``[+]`` / ``[-]``，不用更好看的 ``▶`` / ``▼``：``▶``（U+25B6）**不在 GBK 里**，
#: 而界面字形有一条硬规矩 —— 只用 GBK 里有的（`•` 那个事故见 `tui/console.py`）。
#: ``[+]`` / ``[-]`` 顺带还把"这里能点"写在了脸上，比纯三角形更好认。
FOLD_CLOSED = "[+]"
FOLD_OPEN = "[-]"


@dataclass(frozen=True)
class Folded:
    """同一段内容的**两份写法**：折叠态与展开态。

    ``foldable=False`` 表示"没什么可折"—— 此时两份内容一样，界面**不该给记号**：
    给一个点了没反应的记号，比不给记号更糟。所以"要不要折"这个判断留在 :mod:`tui.render`
    （按内容算），控件只管在两者之间切。
    """

    collapsed: str
    expanded: str
    foldable: bool


#: 回放时**要画**的事件类型。刻意不含过程事件（``model_start`` / ``sys_prompt`` /
#: ``run_end`` / ``session_*``）：时间线要的是"说过什么、做过什么"，不是重演当时的内部节拍。
#:
#: ★ **``tool_start`` 必须在里面**：工具调用的参数只在它身上。少了它，回放出来的工具行
#: 只能写成 ``read · ?`` —— 而参数本来就在会话文件里，没理由丢。
REPLAY_TYPES = frozenset({
    EventType.USER_MESSAGE, EventType.ASSISTANT_MESSAGE,
    EventType.TOOL_START, EventType.TOOL_RESULT,
})


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def fmt_tokens(count: int) -> str:
    """token 数成人化：``12345`` → ``12.3k``；整千不留小数（``60000`` → ``60k``，不是 ``60.0k``）。"""
    if count >= 1_000_000:
        value = count / 1_000_000
        return f"{value:.1f}M" if value % 0.1 else f"{value:.0f}M"
    if count >= 1_000:
        value = count / 1_000
        return f"{value:.0f}k" if value == int(value) else f"{value:.1f}k"
    return str(count)


def cache_rate(cached: int, input_tokens: int) -> float | None:
    """缓存命中率 = **命中 / 输入**。输入为 0（还没调过模型）时给 ``None``。

    ★ 分母是**输入**，不是 ``input + output``。``cached_tokens`` 是 provider 的
    prompt cache 命中数，它本身就是 ``input_tokens`` 的一个子集（见
    `src/core/models.py` 的 ``_usage_from``）；输出 token 永远不可能是缓存命中，
    拿总量当分母只会把这个比例稀释成一个**到不了 100%** 的数 —— 那样标出来的
    "命中率"既不是命中率，也不能跨"输出长短"比较。
    """
    if input_tokens <= 0:
        return None
    return cached / input_tokens


def fmt_duration(ms: int) -> str:
    if ms < 1000:
        return f"{ms}ms"
    if ms < 60_000:
        return f"{ms / 1000:.1f}s"
    minutes, seconds = divmod(ms // 1000, 60)
    return f"{minutes}分{seconds}秒"


def args_brief(args: dict[str, Any], limit: int = 160) -> str:
    """参数压成一行（弹窗、工具块都用它）。"""
    try:
        text = json.dumps(args, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(args)
    return text if len(text) <= limit else text[:limit] + "…"


#: 标记串里的"样式标签"白名单 —— 只认我们自己发出的那些（见 `_STYLE_TAG`）。
_STYLE_TAG = re.compile(r"\\\[|\[/?(?:dim|b|i|u|reverse|yellow|green|red|cyan|magenta|blue|"
                        r"white|black|gray|grey|orange)\b[^\]]*\]")


def strip_markup(text: str) -> str:
    """标记串 → 能直接打给人看的纯文本（`--show-sub` / `--show-tree` 那种出口用）。

    ★ **一次扫完，不能"先去标签、再还原转义"**：内容里的 ``[`` 是被 :func:`escape` 转义成
    ``\\[`` 的，分两步做的话，``abc \\[1]`` 会先被"去标签"削掉 ``[1]``、剩个光秃秃的反斜杠。
    所以这里一个正则同时吃两种：

    - ``\\[``（内容里的方括号）→ 还原成 ``[``；
    - 白名单里的样式标签 → 删掉。
    """
    return _STYLE_TAG.sub(lambda m: "[" if m.group(0) == "\\[" else "", text)


def brief(value: Any, limit: int = LABEL_VALUE_CHARS) -> str:
    """任意一个值 → **一行**、最多 ``limit`` 个字符（尾部 ``…``）。

    换行折成空格：`todo_write` 的 content 本来就是一份多行清单，不折行会把标题行撑成好几行。
    结构化值先转 JSON —— ``str(dict)`` 出来的是 Python 字面量（单引号、``True``），
    既不给人看，也和后端 JSON 对不上。
    """
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, default=str)
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


# ---------------------------------------------------------------------------
# 时间线上的各类块
# ---------------------------------------------------------------------------
def user_block(text: str, *, injected: bool = False) -> str:
    """用户消息。``injected`` 是中途插话的那条（来源不同，标出来）。

    写法：``你 >>> 内容``。**记号反显**（前景背景互换），配合整块的底色带（见 `tui/app.py` 的 CSS）
    一起把"我说的那句"从一堆左对齐的日志里挑出来；箭头 ``>>>`` 比竖线更像"话从这里开始"，
    也不会和工具块的 ``┌`` / ``└`` / ``│`` 混。
    """
    tag = "[dim]（插话）[/dim]" if injected else ""
    return f"[reverse] 你 [/reverse]{tag} [cyan]>>>[/cyan] {escape(text)}"


#: 思考那一行的开头。它和正文一样**压暗 ＋ 斜体**（终端里没有"小一号字体"：
#: 字号由终端和用户设置决定，程序只能改颜色、粗体、斜体、下划线）。
THINK_PREFIX = "think · "


def reasoning_markup(text: str, *, limit: int = FOLD_CHARS) -> Folded:
    """思考过程 → 折叠态 / 展开态。

    超过 ``limit`` 个字符就折起来（默认状态是**关**）：折叠时只显示前 ``limit`` 个字符 ＋ ``…``，
    点一下展开看全文，再点一下收回。短于 ``limit`` 就不折，也不给记号。

    ⚠ **先按原文截、再转义**：反过来会把 ``\\[`` 这样的转义序列切成半截，
    于是屏幕上出现一个没闭合的标记。
    """
    if len(text) <= limit:
        body = f"{THINK_PREFIX}{escape(text)}"
        return Folded(f"[dim italic]{body}[/dim italic]",
                      f"[dim italic]{body}[/dim italic]", False)

    head = escape(text[:limit].rstrip())
    return Folded(
        collapsed=f"[yellow]{FOLD_CLOSED}[/yellow] [dim italic]{THINK_PREFIX}{head}…[/dim italic]",
        expanded=f"[dim]{FOLD_OPEN}[/dim] [dim italic]{THINK_PREFIX}{escape(text)}[/dim italic]",
        foldable=True)


def assistant_block(text: str) -> str:
    """助手说的话 —— **不再转义**：它现在交给 Markdown 渲染（`textual.widgets.Markdown`），
    转义反而会把 `**粗体**`、代码块这些标记打成字面量。

    这条路径只留给测试与"要纯文本"的场合；界面上走的是 `Transcript.add_markdown`。
    """
    return text


@dataclass(frozen=True)
class ToolView:
    """一次工具调用的**展示模型**。

    - ``head``：头两行（``┌ read · a.py`` / ``└ ◆ 完成 · 12ms``），是我们自己的标记；
    - ``body``：结果正文**原文**（不转义 —— 它交给 Markdown 渲染器）；
    - ``hint_closed`` / ``hint_open``：折叠态与展开态各自那一行记号；
    - ``foldable``：有没有正文可折（被拒绝、空结果就没有）。
    """

    head: str
    body: str
    hint_closed: str
    hint_open: str
    foldable: bool


def tool_view(event: Any, *, name: str = "", args: dict[str, Any] | None = None,
              subagent: bool = False) -> ToolView:
    """``tool_result`` → :class:`ToolView`（头 ＋ 正文 ＋ 折叠记号）。

    结果一行里带**怎么结束的 ＋ 花了多久**：轨迹里最常问的两件事就是"它成功了吗"和"它慢在哪"。

    ``name`` / ``args`` 是这次调用（来自 ``tool_start``，实时与回放都拿得到），用来画开头那行
    —— 只留结果的话，事后回看就只剩"◆ 完成 120ms"，看不出到底调了什么。

    **正文默认整块折起来**（哪怕只有两行）：折叠态只给一行"（N 行 · M 字符 · 点开看）"。
    理由是工具结果常是几十行文件内容或日志，摊在时间线上会把对话淹掉；而"这里能点开"这件事
    必须一眼看见，所以提示里带上尺寸，让人自己决定值不值得点。
    """
    outcome = str(event.data.get(DataKey.OUTCOME, ""))
    mark, word = OUTCOME_MARK.get(outcome, ("◆", outcome or "结束"))
    color = "red" if outcome in (Outcome.FAILED, Outcome.TIMEOUT) else (
        "yellow" if outcome in (Outcome.DENIED, Outcome.BLOCKED) else "green")
    head_parts = [f"[dim]└[/dim] [{color}]{mark} {word}[/{color}]"]
    duration = event.data.get(DataKey.DURATION_MS)
    if duration is not None:
        head_parts.append(f"[dim]{fmt_duration(int(duration))}[/dim]")

    truncated_from = event.data.get(DataKey.TRUNCATED_FROM)
    if truncated_from:
        spilled = event.data.get(DataKey.SPILLED_TO)
        tail = f"，全文在 {escape(str(spilled))}" if spilled else ""
        head_parts.append(f"[dim]（结果被截断：原始 {truncated_from} 字符{tail}）[/dim]")

    child = event.data.get(DataKey.CHILD_SESSION)
    if child:
        head_parts.append(f"[dim]（子会话 {escape(str(child))}）[/dim]")

    lines = []
    if name:
        lines.append(tool_head(name, args, subagent=subagent))
    lines.append("[dim] · [/dim]".join(part for part in head_parts if part))
    head = "\n".join(lines)

    message = event.data.get(DataKey.MESSAGE)
    body = str(getattr(message, "content", "") or "")
    if not body.strip():
        # 没有正文（被拒绝、被拦下、空结果）：没什么可折的，也不该给一个点了没反应的记号
        return ToolView(head, "", "", "", False)

    return ToolView(
        head=head,
        body=body,
        hint_closed=(f"[dim]│[/dim] [yellow]{FOLD_CLOSED}[/yellow] "
                     f"[dim]（{body_size(body)} · 点开看）[/dim]"),
        hint_open=f"[dim]│[/dim] [dim]{FOLD_OPEN}[/dim]",
        foldable=True)


def tool_label(name: str, args: dict[str, Any] | None = None) -> str:
    """一次工具调用的**显示名**，如 ``read · src/core/models.py``、``grep · TODO, src``。

    ★ **三档，按这个顺序判**（每档都是 ``<动作> · <参数>``，参数那一段会打上下划线）：

    1. **表里有名字的**（``read_file`` / ``grep`` / ``sandbox_bash`` …）：按表取参数，
       于是 ``write_file`` 永远显示 ``path``（``content`` 可能有几十 KB）、``grep`` 显示
       ``pattern`` 与 ``path``。**"该显示哪个参数"是查表得来的，不是猜的。**
    2. **``mcp__<server>__<tool>`` 那种远端工具**：写成 ``mcp <server> <tool> · <参数>``。
       名字里已经带着出处，所以把出处念出来 —— 否则一屏 ``read_file`` / ``read``
       分不清哪个是本地的、哪个是别人家的。
    3. **剩下的**（``extra_tools`` 塞进来的自定义工具）：``<工具名> · <参数>``。
       参数无从挑，就一个参数报它自己、多个报 JSON（见 :func:`_args_brief`）。

    表里有、但模型没给某个参数时，那个位置显示 ``?``：**不能省略**，否则这一行会变成
    ``list · `` 这种半截话，看起来像界面坏了，而事实是"这次调用没带这个参数"。
    """
    entry = TOOL_LABELS.get(name)
    if entry is not None:
        label, keys = entry
        values = [brief(args[key]) if args and args.get(key) is not None else "?"
                  for key in keys]
        return f"{label} · " + ", ".join(values)

    return f"{_mcp_label(name) or name} · {_args_brief(args)}"


def _mcp_label(name: str) -> str | None:
    """``mcp__fs__read_file`` → ``mcp fs read_file``；不是这个形状就返回 ``None``。

    ★ **从名字里反解，因为名字是构造出来的**：`src/harness/mcp.py` 的 `_mcp_tool_name`
    就是 ``f"mcp__{server}__{tool}".lower()`` 再把非 ``[a-z0-9_]`` 的字符换成 ``_``。
    界面拿到的只有一个名字字符串（回放时更是只从会话文件里读得到名字），所以这是唯一的路。

    切分只切**第一个** ``__``：server 名自己带下划线是常事（``my_server``），
    从左边切才不会把 server 名劈开。代价是 server 名里真的有连续两个下划线时会把
    server 读短一截 —— 那是**显示层面**的取舍，不影响任何调用（工具仍然按原名去调）。
    """
    if not name.startswith(MCP_PREFIX):
        return None
    server, sep, tool = name[len(MCP_PREFIX):].partition("__")
    if not sep or not server or not tool:
        return None                      # `mcp__fs` 这种半截形状：不硬认
    return f"mcp {server} {tool}"


def _args_brief(args: dict[str, Any] | None) -> str:
    """**没有标签**的工具，参数那一小段。

    和权限层那份 ``invocation_summary`` 同一套取舍（一个参数就报它自己、多个报 JSON），
    但**不带括号**：没有标签的工具要和有标签的看起来是同一族
    （``grep · TODO, src`` 对 ``my_tool · x.py``），多一对括号就成了另一种东西。
    """
    if not args:
        return "?"
    if len(args) == 1:
        return brief(next(iter(args.values())))
    return brief(json.dumps(args, ensure_ascii=False, default=str))


def _colored_label(label: str) -> str:
    """给显示名上色 ＋ 给**参数**加下划线：``·`` 前面是动作、后面是宾语。

    分开着色（动作黄）才扫得快；再给参数打上下划线，是因为它是这一行里**最长、最多变**的那段
    （``read · src/core/models.py`` 对 ``read · a.txt``），下划线把它从固定词
    （``read`` / ``write`` / ``grep`` …）里挑出来 —— 眼睛先落在动作上，再顺着下划线看宾语。
    """
    action, sep, rest = label.partition(" · ")
    if not sep:
        return f"[yellow]{escape(label)}[/yellow]"
    return f"[yellow]{escape(action)}[/yellow][dim]{sep}[/dim][u]{escape(rest)}[/u]"


def tool_head(name: str, args: dict[str, Any] | None = None, *, subagent: bool = False) -> str:
    """工具块的开头那行：``┌ read · a.py``（**不带状态**）。

    **实时与回放共用它**：回放时参数来自会话文件里的 ``tool_start``，两边画出来一模一样 ——
    界面不该因为"这是从文件里读回来的"就长得不一样。
    """
    # 委派另加一个记号：它会跑很久、而且花的是自己的钱，值得一眼认出来。
    # （工具名已经是 `task`，所以这个记号是"性质"而不是"名字"。）
    mark = "[magenta]子代理[/magenta] " if subagent else ""
    return f"[dim]┌[/dim] {mark}{_colored_label(tool_label(name, args))}"


def tool_start_line(event: Any) -> str:
    """``tool_start`` → "正在调"的那一行。此时还不知道结果。

    "运行中…"是**临时**的：结果回来时整块会被重画成 :func:`tool_view` 给的样子。
    （早期版本把这一行原样留在成品里，于是那句"运行中…"永久挂在了已完成的调用上。）
    """
    return (tool_head(str(event.data.get("name", "?")), dict(event.data.get("args") or {}),
                      subagent=event.data.get(DataKey.TOOL_TYPE) == "subagent")
            + " [dim]运行中…[/dim]")


def body_size(text: str) -> str:
    """正文的一行尺寸说明，如 ``12 行 · 480 字符``。

    折叠态靠它告诉人"值不值得点开"：只有"（点开看）"而不知道多大，等于让人盲点。
    """
    return f"{len(text.rstrip().splitlines())} 行 · {len(text)} 字符"


def unfinished_block(name: str, args: dict[str, Any] | None = None, *,
                     subagent: bool = False) -> str:
    """一次**没有结果**的调用（当时被打断 / 崩了）：那一行留着并说明。

    那一块**确实发生过**（工具真的被调了，事件也落盘了），让它凭空消失比留一行"○ 打断"更糟。
    """
    return (f"{tool_head(name, args, subagent=subagent)}\n"
            f"[dim]└[/dim] [yellow]○ 打断[/yellow] "
            f"[dim]（没有结果：那一轮当时就停在这里）[/dim]")


def gutter(text: str, *, marker: str = "") -> str:
    """给多行内容加左边的竖线（并可选地在第一行放一个记号）。**内容是原文，这里负责转义。**"""
    body = escape(text)
    prefix = f"[yellow]{marker}[/yellow] " if marker else ""
    return f"[dim]│[/dim] {prefix}" + body.replace("\n", "\n[dim]│[/dim] ")


# ---------------------------------------------------------------------------
# 回放：**一整份事件文件 → 时间线上的块**（纯函数）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Block:
    """时间线上一块的内容（**不含控件**）。

    ``kind`` 是块的性质，同时就是 CSS 类名（见 `tui/app.py` 的 ``CSS``）。后面四个字段
    说明"这一块该用哪种控件画"，四选一：

    - ``markdown=True``：``text`` 是 **Markdown 原文**（助手回答：转义会把 ``**粗体**``
      打成字面量，所以这一档必须交给 Markdown 解析器）；
    - ``folded`` 非空：**可折叠的标记文本**（思考过程：两份写法已由 :func:`reasoning_markup` 算好）；
    - ``tool`` 非空：工具块（头 ＋ 可折叠的 Markdown 正文）；
    - 都不是：``text`` 是普通标记串（用户消息、说明、分隔线那一类）。

    **纯数据、不 import 界面框架**，所以"事件序列 → 时间线"这件事不起终端就能测。
    """

    kind: str
    text: str = ""
    markdown: bool = False
    folded: Folded | None = None
    tool: ToolView | None = None


@dataclass(frozen=True)
class Replay:
    """一次回放的结果：**要画的那些块** ＋ 顺手算出来的账。

    账（步数 / 轮数 / 用量）是回放这一趟的副产品：界面要拿它把状态栏接到"历史累计"上，
    子会话视图要拿它在抬头写一行 —— 两处都得从同一份事件里数，所以别让调用方再走一遍。
    """

    blocks: tuple[Block, ...] = ()
    steps: int = 0
    turns: int = 0
    usage: Usage | None = None


def replay(events: Sequence[Any], *, fold_chars: int = FOLD_CHARS) -> Replay:
    """一整份事件文件 → 时间线上的块。

    ★ 读的是**整份事件文件**（``Session.events()`` / ``SessionStore.events_of()``），
    不是 ``load_ui_events()``：后者只给"进历史"的三类内容事件，而工具调用的**参数**
    在 ``tool_start`` 里 —— 少了它，回放出来的工具行只能写成 ``read · ?``。
    参数本来就在文件里，没理由丢。

    画法与实时那条路**一一对应**：``tool_start`` 与 ``tool_result`` 按 ``span`` 配对，
    配出来的那一块和当时屏幕上看到的（以及结果回来时重画的那一块）完全一样。
    过程事件（``model_start`` / ``sys_prompt`` / ``run_end`` / ``session_*``）不画：
    时间线要的是"说过什么、做过什么"，不是重演当时的内部节拍 —— 但 ``run_end`` 的
    **用量与轮数**要收（那是权威总量：含压缩摘要与子代理的消耗），顺手在同一趟里做了。

    没等到结果的调用（当时被打断 / 崩了）也留一块（见 :func:`unfinished_block`）。
    """
    blocks: list[Block] = []
    pending: dict[str, tuple[str, dict[str, Any], bool]] = {}
    steps = 0
    turns = 0
    usage: Usage | None = None

    for event in events:
        if event.type == EventType.USER_MESSAGE:
            text = str(getattr(event.data.get(DataKey.MESSAGE), "content", "") or "")
            if text:
                blocks.append(Block(KIND_USER, user_block(text)))
        elif event.type == EventType.ASSISTANT_MESSAGE:
            message = event.data.get(DataKey.MESSAGE)
            # ★ 回放时思考内容**也在**：它不在事件流里（`reasoning` 是 live-only、不落盘），
            #   而是在这条助手消息的 `reasoning_content` 上（落盘时一起写下去了）。
            #   实时那次看的是增量，回放这次看的是整段 —— 同一份内容的两条来源。
            reasoning = str(getattr(message, "reasoning_content", "") or "")
            if reasoning:
                blocks.append(Block(KIND_REASONING,
                                    folded=reasoning_markup(reasoning, limit=fold_chars)))
            text = str(getattr(message, "content", "") or "")
            if text:
                blocks.append(Block(KIND_ASSISTANT, text, markdown=True))
        elif event.type == EventType.TOOL_START:
            pending[str(event.span)] = (
                str(event.data.get("name", "?")), dict(event.data.get("args") or {}),
                event.data.get(DataKey.TOOL_TYPE) == "subagent")
            steps += 1                        # 回放也数：累计步数要接得上
        elif event.type == EventType.TOOL_RESULT:
            name, args, subagent = pending.pop(str(event.span), ("", {}, False))
            blocks.append(Block(KIND_TOOL, tool=tool_view(event, name=name, args=args,
                                                          subagent=subagent)))
        elif event.type == EventType.RUN_END:
            got = event.data.get(DataKey.USAGE)
            if isinstance(got, Usage):
                usage = got if usage is None else usage + got
            turns += event.turn or 0

    for name, args, subagent in pending.values():
        blocks.append(Block(KIND_TOOL, unfinished_block(name, args, subagent=subagent)))

    return Replay(tuple(blocks), steps=steps, turns=turns, usage=usage)


def block_text(block: Block) -> str:
    """一块 → 纯文本（``--show-sub`` 那种不进界面的出口用）。

    跟着**屏幕的默认状态**走：工具正文是折着的（只给一行尺寸提示），思考块超过阈值也只给
    开头一段。不这么做的话，一份时间线的纯文本会被工具结果（几十行文件内容或日志）淹掉 ——
    那正是界面上把它们默认折起来的原因。
    """
    if block.tool is not None:
        parts = [strip_markup(block.tool.head)]
        if block.tool.foldable:
            parts.append(strip_markup(block.tool.hint_closed))
        return "\n".join(parts)
    if block.folded is not None:
        return strip_markup(block.folded.collapsed)
    if block.markdown:
        return block.text                    # Markdown 原文：不是标记串，别再"去标记"
    return strip_markup(block.text)



def notice_block(text: str) -> str:
    """一条压暗的说明（启动信息、模式切换、插话已投递一类）。"""
    return f"[dim]{escape(text)}[/dim]"


def divider_block(text: str = "") -> str:
    """一条分隔线。用来标"上面是回放的历史、下面是这一次"。"""
    line = "─" * 12
    return f"[dim]{line} {escape(text)} {line}[/dim]" if text else f"[dim]{line}[/dim]"


def run_end_block(event: Any) -> str:
    """``run_end`` → 一条压暗的分隔说明。"""
    reason = str(event.data.get(DataKey.STOP_REASON, ""))
    word = STOP_WORD.get(reason, reason or "结束")
    parts = [f"[dim]本轮结束：{word}[/dim]"]
    usage = event.data.get(DataKey.USAGE)
    if isinstance(usage, Usage):
        parts.append(f"[dim]{usage_brief(usage)}[/dim]")
    duration = event.data.get(DataKey.DURATION_MS)
    if duration is not None:
        parts.append(f"[dim]{fmt_duration(int(duration))}[/dim]")
    if event.turn:
        parts.append(f"[dim]{event.turn} 轮[/dim]")
    return "[dim]·[/dim] " + " [dim]·[/dim] ".join(parts)


def compaction_block(event: Any) -> str:
    """``compaction`` → 说明"历史被改写了"。**必须显示**：不显示的话，界面上会莫名其妙
    少掉一段上下文，而用户看不出是压缩干的。"""
    strategy = str(event.data.get("strategy", ""))
    before = event.data.get("before_tokens", 0)
    after = event.data.get("after_tokens", 0)
    if strategy == "summarize":
        extra = f"摘掉 {event.data.get('dropped_messages', 0)} 条，留给摘要模型"
    else:
        extra = f"清掉 {event.data.get('cleared_chars', 0)} 字符的旧工具结果"
    return (f"[dim]※ 压缩上下文（{escape(strategy or '?')}）："
            f"约 {before} → {after} token，{extra}[/dim]")


def error_block(event: Any) -> str:
    source = str(event.data.get(DataKey.ERROR_SOURCE, ""))
    message = str(event.data.get("message", ""))
    return f"[red]▲ 出错（{escape(source or '?')}）[/red] {escape(message)}"


def approval_head(request: Any) -> str:
    """审批弹窗的标题行：**要批的到底是哪一次调用**。

    用和工具块同一个显示名（:func:`tool_label`）—— 弹窗上认一次、时间线上再看一次，
    两处长得不一样的话，人得在脑子里做一次映射才能确认"批的就是刚才那个"。
    """
    call = request.call
    return _colored_label(tool_label(str(getattr(call, "name", "?")),
                                     dict(getattr(call, "args", {}) or {})))


def approval_facts(request: Any) -> list[str]:
    """审批弹窗的几行事实：风险、为什么问、不回答会怎样、参数。"""
    risk = RISK_WORD.get(str(request.risk), str(request.risk))
    consequence = "放行" if request.proposed else "拒绝"
    left = max(0, int((request.expires_ts - now_ms()) / 1000))
    return [
        f"[b]风险等级[/b]　{escape(risk)}",
        f"[b]为什么问[/b]　{escape(str(request.reason))}",
        f"[b]不回答[/b]　{left} 秒后按默认值[b]{consequence}[/b]（{escape(risk)}级工具的出厂默认）",
        f"[b]参数[/b]　　{escape(args_brief(dict(getattr(request.call, 'args', {}) or {}), 400))}",
    ]


# ---------------------------------------------------------------------------
# 状态栏
# ---------------------------------------------------------------------------
#: 界面在干什么（状态栏上那个词）。
PHASE_IDLE = "空闲"
PHASE_MODEL = "请求模型"
PHASE_TOOL = "执行工具"
PHASE_APPROVAL = "等待审批"
PHASE_DONE = "出错"


@dataclass
class StatusState:
    """状态栏要显示的全部东西。**没有行为**，所以 :func:`status_line` 是可测的纯函数。"""

    mode: str = "ask"
    model: str = "?"
    session: str = ""
    workspace: str = ""
    phase: str = PHASE_IDLE
    #: **累计**轮次（模型调用轮数）与**累计**步数（工具调用次数）—— 跨 run 累加，
    #: 不是"这一轮的第几轮"：会话接着聊、接着跑，这两个数只会涨。
    turns: int = 0
    steps: int = 0
    usage: Usage | None = None
    #: **当前上下文窗口占了预算多少**（`src/core/context.py` 的估算器给的数）
    #: 与预算本身（`cfg.max_context_tokens`）。``max`` 为 0 时不显示这一段。
    context_tokens: int = 0
    context_max: int = 0
    running: bool = False


def context_brief(tokens: int, max_tokens: int) -> str:
    """``上下文 23%（14.1k/60k）`` —— 上下文窗口占用。

    百分比是分母（``cfg.max_context_tokens``）算出来的**粗估**：数本身来自
    `src/core/context.py` 的 :func:`~src.core.context.estimate_tokens` —— 最近一次模型调用的
    真实计数（锚点）＋ 它之后新增内容的字符估算。所以它不是"模型的精确计数"，
    而是"离压缩阈值还有多远"的指示器（`compact_threshold` 默认 0.75，到那儿就该压了）。
    """
    ratio = tokens / max_tokens if max_tokens > 0 else 0.0
    return f"上下文 {ratio * 100:.0f}%（{fmt_tokens(tokens)}/{fmt_tokens(max_tokens)}）"


def usage_brief(usage: Usage) -> str:
    """``输入 12.4k tok · 输出 812 tok · 思考 65 tok · 缓存命中 87%`` —— token 那一段。

    ★ 命中率的分母见 :func:`cache_rate`（是**输入**，不是总量）；**输入为 0 时不显示这一项**，
    而不是显示 ``0%``：那会让人以为"一次都没命中"，而事实是"还没有过一次调用"。
    """
    parts = [f"输入 {fmt_tokens(usage.input_tokens)} tok",
             f"输出 {fmt_tokens(usage.output_tokens)} tok",
             f"思考 {fmt_tokens(usage.reasoning_tokens)} tok"]
    rate = cache_rate(usage.cached_tokens, usage.input_tokens)
    if rate is not None:
        parts.append(f"缓存命中 {rate * 100:.0f}%")
    return " [dim]·[/dim] ".join(parts)


def status_line(state: StatusState) -> str:
    """状态栏那一行：

        模式 | 模型 | 会话 | 累计 x 轮 · 共 xx 步 | 输入 … tok · 输出 … tok · 思考 … tok · 缓存命中 …% | 上下文 …% | 空闲

    **大段之间用 ``|``，段内用 ``·``**：段是"互不相干的几件事"（谁在跑 / 跑了多少 / 烧了多少 /
    上下文多满 / 在干什么），用同一种分隔符串起来会读成一长串。

    **两个计数器都是累计的**（跨 run 累加）：``轮`` = 模型调用的轮数，``步`` = 工具调用次数。
    会话接着聊的时候，"现在第几轮"会从 1 重新数，而"累计跑了多少"才是想看的那个数。

    还没调过模型时不显示 token 那一段（全 0 的一串没有信息，只会把状态栏撑长）；
    上下文那一段要 ``context_max`` 有值才显示（没配预算就没有百分比可言）。
    """
    parts = [
        f"[b]{escape(state.mode)}[/b] 模式",
        escape(state.model),
        f"会话 {escape(state.session or '-')}",
        f"累计 {state.turns} 轮 [dim]·[/dim] 共 {state.steps} 步",
    ]
    if state.usage is not None and state.usage.total_tokens:
        parts.append(usage_brief(state.usage))
    if state.context_max > 0:
        parts.append(context_brief(state.context_tokens, state.context_max))
    parts.append(f"[b]{escape(state.phase)}[/b]")
    return "[dim] | [/dim]".join(parts)
