import json
import os
import time
from dataclasses import replace, dataclass

from src.core.events import CompactionStrategy
from src.core.message import AnyMessage, AIMessage, Messages, ToolMessage, HumanMessage, Usage, SystemMessage

#: 粗估用的换算：每多少字符算一个 token（中英混合的保守值）。
CHARS_PER_TOKEN = 4

#: 每条消息的固定开销（role、分隔符、包装）。消息越多，这部分越不能忽略。
MESSAGE_OVERHEAD_TOKENS = 8

DEFAULT_MAX_CONTEXT_TOKENS = 128_000
#: 用到预算的这个比例就先试一档轻量压缩（软阈值）。
DEFAULT_COMPACT_THRESHOLD = 0.75
#: 压缩时"最近 N 条消息"一律不动。工具结果是被消费过就没用的东西，
#: 越久远的越可以清 —— 结论通常已经沉淀进后面的 AI 消息里了。
DEFAULT_KEEP_RECENT = 8
#: 太短的工具结果不值得清：清了省不下 token，反而白丢信息。
DEFAULT_TOOL_RESULT_MIN_CHARS = 200

#: 单个记忆文件的上限。实测过的教训：2MB 的 AGENTS.md 会被整体塞进 prompt（≈ 52 万 token）。
DEFAULT_MEMORY_MAX_FILE_BYTES = 32 * 1024
#: 全部记忆文件的合计上限。
DEFAULT_MEMORY_MAX_TOTAL_BYTES = 64 * 1024

# ---------------------------------------------------------------------------
# 估算
# ---------------------------------------------------------------------------
def estimate_text_tokens(text: str) -> int:
    """纯文本的粗估。给"固定开销"用（system prompt / 记忆文件）。"""
    if not text:
        return 0
    return len(text) // CHARS_PER_TOKEN

def _message_tokens(message: AnyMessage) -> int:
    """一条消息的粗估：正文 ＋ **工具调用的参数** ＋ 固定开销。

    工具调用的参数必须算进去。老实现只数 ``content``，于是一条
    ``write_file(path=…, content=<4000 字符>)`` 被算成 8 token（复核时实测：真实约 1000）。
    """
    chars = len(message.content or "")
    for call in getattr(message, "tool_calls", None) or ():
        chars += len(json.dumps(call.args or {}, ensure_ascii=False, default=str))
    if isinstance(message, AIMessage) and message.reasoning_content:
        chars += len(json.dumps(message.reasoning_content, ensure_ascii=False, default=str))
    return chars // CHARS_PER_TOKEN + MESSAGE_OVERHEAD_TOKENS

def estimate_tokens(messages: Messages, tools_schema:list | None = None) -> int:
    """估算这串历史会占多少 token。

    **一级（锚点）**：从后往前找最近一条**带 ``usage`` 的 AI 消息**。它的 ``total_tokens``
    是 provider 对**那一次真实请求**的计数 —— 而那次请求里就带着 system prompt 与全部工具
    schema，所以**固定开销已经含在里面了，不要再加一遍**。
    （用 ``total`` 而不是 ``input``：下一次请求的输入本来就包含上一次的输出。）

    **二级（粗估）**：锚点之后新增的消息按字符数粗估。

    **没有锚点时**（会话第一次模型调用之前、或压缩刚把 ``usage`` 清掉之后）：
    全部走粗估，**并且要把 ``fixed_overhead_tokens`` 加上** —— 只有这时候固定开销才真的不在里面。
    """

    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        usage = getattr(message, "usage", None)
        if isinstance(message, AIMessage) and usage is not None and usage.input_tokens:
            base = usage.total_tokens
            rough = sum(_message_tokens(m) for m in messages[index + 1:])
            return base + rough
    return sum(_message_tokens(m) for m in messages) + estimate_text_tokens(str(tools_schema))

# ---------------------------------------------------------------------------
# 项目记忆（AGENTS.md / CLAUDE.md 惯例）
# ---------------------------------------------------------------------------
def _walk_memory_paths(cwd: str, names) -> list[str]:
    """从 ``cwd`` 逐级向上找记忆文件，返回**从 cwd 往上**的顺序（调用方按需反转）。"""
    found: list[str] = []
    directory = os.path.abspath(cwd)
    while True:
        for name in names:
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate):
                found.append(candidate)
                break                      # 同一目录里有多个候选时只取第一个
        parent = os.path.dirname(directory)
        if parent == directory:
            break
        directory = parent
    return found


def memory_source_paths(cwd: str, *, names=("AGENTS.md", "CLAUDE.md")) -> list[str]:
    """这次 run 的 prompt 会由**哪些文件**拼成 —— 按它们真正生效的顺序（根目录在前）。

    单独暴露出来是给 trace 用的：``sys_prompt`` 事件要记 ``memory_sources``，而事后从拼好的
    文本里**认不出路径**（内容在，出处没了）。

    只走目录、**不读文件内容**（只有 ``isfile`` 检查），所以和 ``load_memory_files`` 分开调
    也不贵 —— 后者那句"每次 run 只调一次"说的是**读盘**，不是这里。
    """
    return list(reversed(_walk_memory_paths(cwd, names)))


def load_memory_files(cwd: str, *, names=("AGENTS.md", "CLAUDE.md"),
                      max_file_bytes: int = DEFAULT_MEMORY_MAX_FILE_BYTES,
                      max_total_bytes: int = DEFAULT_MEMORY_MAX_TOTAL_BYTES) -> str:
    """从 ``cwd`` 逐级向上收集项目记忆文件：外层在前、内层在后（内层更具体、优先级更高）。

    这是各家 harness 的共同惯例：把"项目怎么构建、规范是什么、别碰哪些目录"写进仓库里的
    markdown，agent 每次启动自动读入 —— **记忆放在文件系统里，而不是模型里**。

    三条健壮性约定（老实现一条都没有，复核实测过后果）：

    - **每个文件有大小上限**，超了就截断并留一句说明（实测：2MB 的 AGENTS.md 会被整体塞进
      prompt ≈ 52 万 token）；
    - **读不了就跳过**，并在结果里留一行说明，而不是让 ``UnicodeDecodeError`` 把整个 run 干掉；
    - **这个方法每次 run 只调一次**：它要读磁盘，而 ``ContextManager.assemble`` 是纯函数、
      根本不碰磁盘。
    """
    found = _walk_memory_paths(cwd, names)

    chunks: list[str] = []
    used = 0
    for path in reversed(found):           # 根目录的在前，越靠近 cwd 的越后（后者覆盖前者）
        try:
            with open(path, "rb") as handle:
                raw = handle.read(max_file_bytes + 1)
        except OSError as exc:
            chunks.append(f"<memory source={path!r} skipped={f'读不了: {exc}'!r} />")
            continue

        truncated = len(raw) > max_file_bytes
        raw = raw[:max_file_bytes]
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="replace")
            text += "\n…[这个文件不是合法 UTF-8，已用替换字符读入]"
        if truncated:
            text += f"\n…[记忆文件超过 {max_file_bytes} 字节，已截断]"

        if used + len(text) > max_total_bytes:
            keep = max(0, max_total_bytes - used)
            text = text[:keep] + f"\n…[记忆总量超过 {max_total_bytes} 字节，已截断]"
        used += len(text)
        chunks.append(f"<memory source={path!r}>\n{text.strip()}\n</memory>")

    return "\n\n".join(chunks)

# ---------------------------------------------------------------------------
# 轻量压缩
# ---------------------------------------------------------------------------

def clear_old_tool_results(messages: Messages, keep_recent: int = DEFAULT_KEEP_RECENT,
                           *, min_chars: int = DEFAULT_TOOL_RESULT_MIN_CHARS,
                           placeholder: str = "[旧工具结果已清理，共 {n} 字符]") -> tuple[Messages, tuple[int, ...]]:
    """把**久远**的工具结果正文换成占位符，返回 ``(新历史, 被清理的下标)``。

    - **保留消息结构**：条数不动、顺序不动、``tool_call_id`` / ``name`` 不动。于是永远不会
      切出"孤儿工具结果"。

    - **保留 ``is_error``**：老实现重建 ``ToolMessage`` 时把它丢了，
      模型会以为失败过的调用是成功的，然后基于错误前提继续干活。

    - **只清比 ``min_chars`` 长的**：太短的清了省不下 token，反而白丢信息。

    返回的下标是给上层用的：``core`` 不认识 ``seq``（那是文件层的事），
    上层靠它把"清了哪几条"写进 ``compaction`` 事件。
    """
    boundary = max(0, len(messages) - max(0, keep_recent))
    cleared: list[int] = []
    out: Messages = []
    for index, message in enumerate(messages):
        content = message.content or ""
        if index < boundary and isinstance(message, ToolMessage) and len(content) > min_chars:
            cleared.append(index)
            out.append(replace(message, content=placeholder.format(n=len(content))))
        else:
            out.append(message)
    return out, tuple(cleared)

# ---------------------------------------------------------------------------
# AI 总结压缩（summarize）
# ---------------------------------------------------------------------------
_SUMMARIZE_PROMPT = """\
你是对话压缩器。把下面这段 Agent 工作记录压缩成一份"前情提要"，供同一个 Agent \
继续工作使用。必须保留：用户的原始目标与约束、已完成的事与关键结论、正在进行的事、\
尚未解决的问题、重要的文件路径/命令/数据。省略：寒暄、失败后已被纠正的弯路细节、\
工具输出的原文（只留结论）。用条目式中文输出。"""


def _render_transcript(messages: Messages) -> str:
    """把要送进摘要的历史转成一段文字（角色 ＋ 工具调用 ＋ 正文）。"""
    lines = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            calls = "; ".join(f"{c.name}({c.args})" for c in message.tool_calls)
            lines.append(f"[assistant 调用工具] {calls}")
            if message.content:
                lines.append(f"[assistant] {message.content}")
        else:
            lines.append(f"[{message.role}] {message.content}")
    return "\n".join(lines)

def _safe_cut(messages: Messages, keep_recent: int) -> int:
    """返回切分点 ``cut``：``messages[cut:]`` 保留、``messages[:cut]`` 送摘要。

    向前回退直到保留段不以 ``ToolMessage`` 开头 —— 否则这些结果对应的
    ``AIMessage(tool_calls)`` 被切走了，provider 会拒绝这种"孤儿工具结果"。
    """
    cut = max(0, len(messages) - max(0, keep_recent))
    while cut > 0 and isinstance(messages[cut], ToolMessage):
        cut -= 1
    return cut

# ---------------------------------------------------------------------------
# 压缩结果
# ---------------------------------------------------------------------------

@dataclass
class CompactionResult:
    """一次压缩的结果 —— 由上层写成 ``compaction`` 事件（core 不碰事件）。

    两种策略，字段各取所需：

    - ``clear_tool_results``：``cleared_call_ids``（被清理工具结果的 ``tool_call_id``，
      消息自己带着、文件里也有 —— 最稳定的 key），重放器据此把对应的 ``tool_result``
      正文换回占位符；
    - ``summarize``：``recap``（前情提要）、``kept_count``（保留最后几条）、
      ``dropped_messages``（摘掉了几条）、``summarizer_model`` / ``summarizer_usage``
      （摘要调用花在哪个模型上、多少 token）。

    ⚠ **``after_tokens`` 可能等于 ``before_tokens``，这是正常的**：估算的一级近似用的是
    "最近一条带 usage 的 AI 消息"这个**锚点**，它是 provider 对那次请求的计数；而被清掉的
    旧工具结果本来就在那次请求**之前**，锚点里没重复计它们 —— 于是"清掉的东西"在这两个数
    上看不出来。真正的效果由**下一次模型调用**的 ``usage`` 反映（锚点每轮刷新）。

    所以调用方**不要在同一轮里**拿 ``before/after`` 判断"压得够不够"。判定顺序是：
    先看 ``over_budget`` → 试压 → **压得动就继续跑**（结果非空），
    压不动（``compact`` / ``summarize`` 都返回 ``None``）且仍超限，才明确失败。
    """

    strategy: str
    messages: Messages
    before_tokens: int
    after_tokens: int
    # ---- clear_tool_results 专用 ----
    cleared_call_ids: tuple[str, ...] = ()
    cleared_chars: int = 0
    saved_tokens_estimate: int = 0
    # ---- summarize 专用 ----
    recap: HumanMessage | None = None
    kept_count: int = 0
    dropped_messages: int = 0
    summarizer_model: str | None = None
    summarizer_usage: Usage | None = None
    # ---- 通用 ----
    duration_ms: int = 0

# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------
@dataclass
class ContextManager:
    """上下文预算与装配策略。挂到 Agent 上即生效（不挂 = 全量历史直发）。"""

    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    #: 软阈值：用到预算的这个比例就先试一档轻量压缩
    compact_threshold: float = DEFAULT_COMPACT_THRESHOLD
    #: 压缩时最近 N 条消息一律不动
    keep_recent: int = DEFAULT_KEEP_RECENT
    #: harness 注入的附加段（技能清单等）
    extra_context: str = ""

    # ---- 装配 ----

    def assemble(self, system_prompt: str, messages: Messages, *, memory: str = "") -> Messages:
        """拼出真正发给模型的列表：``[SystemMessage] + 历史``。

        **纯函数**（不读磁盘）：记忆由调用方**每次 run 加载一次**再传进来（``load_memory_files``）。
        老实现的 ``assemble`` 自己读盘，于是每轮都重读一遍 —— 把 I/O 从函数里拿掉，
        这个错就再也不可能犯。
        """
        parts = [system_prompt.strip()] if system_prompt and system_prompt.strip() else []
        if memory and memory.strip():
            parts.append("# 项目记忆\n" + memory.strip())
        if self.extra_context and self.extra_context.strip():
            parts.append(self.extra_context.strip())
        text = "\n\n".join(parts)
        return ([SystemMessage(content=text)] if text else []) + list(messages)

    # ---- 判断 ----
    def should_compact(self, messages: Messages, tools_schema : list) -> bool:
        """到**软阈值**了吗 —— 到了就先试一档轻量压缩。"""
        return estimate_tokens(messages, tools_schema) > self.max_context_tokens * self.compact_threshold


    # ---- 压缩 ----
    def compact(self, messages: Messages, tools_schema:list) -> CompactionResult | None:
        """轻量压缩一档：清理久远的工具结果正文。

        返回 ``None`` 表示**没东西可压**（没有够长、够旧的工具结果）——
        调用方据此判断：这时候还超限，就只能明确失败了。
        """
        started = time.monotonic()
        cleared, indices = clear_old_tool_results(messages, self.keep_recent)
        if not indices:
            return None

        cleared_chars = sum(len(messages[index].content or "") for index in indices)
        return CompactionResult(
            strategy=CompactionStrategy.CLEAR_TOOL_RESULTS,
            messages=cleared,
            before_tokens=estimate_tokens(messages, tools_schema),
            after_tokens=estimate_tokens(cleared, tools_schema),
            cleared_call_ids=tuple(messages[index].tool_call_id for index in indices),
            cleared_chars=cleared_chars,
            saved_tokens_estimate=cleared_chars // CHARS_PER_TOKEN,
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    def summarize(self, messages: Messages, summarizer) -> CompactionResult | None:
        """AI 总结压缩一档：把早期历史摘要成一段 ``recap``，旧消息丢掉。

        - **``summarizer`` 必须是"不绑工具"的模型**（由调用方传入）：绑了工具的模型可能
          返回 ``tool_calls`` 而不是摘要，此时 ``content`` 是空的；

        - **失败或摘要为空 → 原样返回 ``None``，历史一个字节都不动**（宁可不压，也不拿
          空壳换掉真历史）；

        - **非流式、不推 ``text_delta``**（这里只用 ``invoke``）。
        """
        started = time.monotonic()
        cut = _safe_cut(messages, self.keep_recent)
        if cut <= 1:
            return None                       # 没什么可压的
        head, tail = messages[:cut], messages[cut:]

        try:
            summary = summarizer.invoke([
                SystemMessage(content=_SUMMARIZE_PROMPT),
                HumanMessage(content=_render_transcript(head)),
            ])
        except Exception:
            return None                       # 摘要失败：原样返回
        if not (summary.content or "").strip():
            return None                       # 空摘要视同失败

        recap = HumanMessage(content="[前情提要——早前对话已压缩，以下是摘要]\n" + summary.content)
        # 摘要把早期历史摘掉了，保留段里 AI 消息的 ``usage`` 记的是"压缩前"的账，
        # 必须清空 —— 否则预算估算持续高估 → 每轮都压缩一次（老项目 resume 后复活的根因）。
        fresh_tail = [replace(m, usage=Usage()) if isinstance(m, AIMessage) else m for m in tail]

        new_messages: Messages = [recap] + fresh_tail
        return CompactionResult(
            strategy=CompactionStrategy.SUMMARIZE,
            messages=new_messages,
            before_tokens=estimate_tokens(messages),
            after_tokens=estimate_tokens(new_messages),
            recap=recap,
            kept_count=len(fresh_tail),
            dropped_messages=cut,
            summarizer_model=getattr(summarizer, "model", type(summarizer).__name__),
            summarizer_usage=summary.usage,
            duration_ms=int((time.monotonic() - started) * 1000),
        )