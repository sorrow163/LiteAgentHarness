import json
import os
import time
import uuid
from collections.abc import Iterable
from dataclasses import replace

from typing import Any

from src.core.errors import SessionError
from src.core.events import Event, EventType, DataKey, CompactionStrategy, SessionHead, dumps, loads
from src.core.message import Messages, ToolMessage, Usage, AIMessage


# ---------------------------------------------------------------------------
# 重放规则（运行时与重放**共用这一份**）
# ---------------------------------------------------------------------------

def _apply_to_window(window: Messages, event: Event) -> Messages:
    """把一条事件作用到上下文窗口上 —— 重放规则的**唯一实现**。

    运行时（``Session.write_event``）与重放（``replay_messages``）都走它，
    所以"压缩后重放出的历史 == 当时真正发给模型的历史"不是靠两处代码碰巧一致。
    """
    if event.type in EventType.HISTORY:
        # 子文件的根 ``tool_result``（tool_type=subagent）是"委派的括号"，**不带 message** ——
        # 它不进子代理自己的历史；只有带 message 的历史事件才追加。
        message = event.data.get(DataKey.MESSAGE)
        if message is not None:
            return list(window) + [message]

    if event.type == EventType.COMPACTION:
        strategy = event.data.get("strategy")
        if strategy == CompactionStrategy.CLEAR_TOOL_RESULTS:
            cleared = set(event.data.get("cleared_call_ids", ()))
            return [
                replace(message, content="[旧工具结果已清理，共 {n} 字符]".format(n=len(message.content or "")))
                if isinstance(message, ToolMessage) and message.tool_call_id in cleared else message
                for message in window
            ]
        if strategy == CompactionStrategy.SUMMARIZE:
            # 丢掉最旧的 N 条、换成 recap；保留段里 AI 消息的 usage 是"压缩前"的陈旧账，必须清空。
            recap = event.data.get("recap")
            kept_count = event.data.get("kept_count", 0)
            kept = list(window[-kept_count:]) if kept_count else []
            kept = [replace(m, usage=Usage()) if isinstance(m, AIMessage) else m for m in kept]
            return ([recap] if recap is not None else []) + kept
    return window

def _trim_incomplete_tail(messages: Messages) -> Messages:
    """裁掉末尾"崩在半轮"的尾巴：带 ``tool_calls`` 但结果没写全的 AI 消息、不完整的工具结果。

    恢复点必须是**合法的消息边界** —— 否则把一条"要了工具但没结果"的 AI 消息喂给模型，provider 会直接拒绝。
    """
    trimmed = list(messages)
    while trimmed:
        last = trimmed[-1]
        if getattr(last, "tool_calls", None):
            trimmed.pop()                    # AI 消息要了工具，结果没齐 → 裁掉这半轮
            continue
        if getattr(last, "role", "") == "tool":
            i = len(trimmed) - 1
            while i >= 0 and getattr(trimmed[i], "role", "") == "tool":
                i -= 1
            need = {c.id for c in getattr(trimmed[i], "tool_calls", [])} if i >= 0 else set()
            got = {m.tool_call_id for m in trimmed[i + 1:]}
            if need and need != got:
                del trimmed[i:]              # 工具结果凑不齐一整批 → 整半轮裁掉
                continue
        break
    return trimmed

def replay_messages(events: Iterable[Event]) -> Messages:
    """从事件流重放历史（含尾裁剪）。"""
    window: Messages = []
    for event in events:
        window = _apply_to_window(window, event)
    return _trim_incomplete_tail(window)


class Session:
    """一个会话 = 一个 JSONL 文件 + **内存里的最新上下文窗口**。

    - 首次创建：``messages`` 是**空列表**；
    - 重新打开：基于文件重放出最新的上下文窗口（含压缩与尾裁剪）。
    """

    def __init__(self, path: str, session_id: str, parent: int | str = -1, meta: dict[str, Any] | None = None):
        self.path = path
        self.id = session_id
        self.parent = parent
        self.meta = dict(meta or {})
        self.messages: Messages = []          # 最新上下文窗口（首次创建 = 空）
        self._seq = 0                          # 下一次写入用的文件内自增序号
        self._has_head = False

# ---- 写（非惰性） ----

    def _ensure_head(self) -> None:
        """保证文件第一行是文件头。首次写盘时补上；``create`` 会先调它（空会话也有头）。"""
        if self._has_head:
            return
        head = SessionHead(session_id=self.id, parent=self.parent, meta=self.meta)
        missing = head.missing_meta()
        if missing:
            raise SessionError(
                f"会话 {self.id!r} 的 meta 缺评估必需的键: {sorted(missing)}，"
                "agent / workspace / model / provider / harness / config 缺一不可）"
            )
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write(dumps(head) + "\n")
        self._has_head = True

    def write_event(self, event: Event) -> None:
        """把一条事件**立刻**写进文件，并同步更新内存里的上下文窗口。

        ``seq`` 由会话层在此处按文件内自增赋值（loop 不负责 seq）。
        """
        # 打字机增量只实时、**不落盘**，要在这里跳过，而不是在
        # 上层跳过 —— 因为"落盘 = 不在 LIVE_ONLY 里"是写文件这一方的职责。
        # 跳过要放在 seq 自增**之前**，否则文件里会跳号，而跳号被约定为"丢行"。
        if event.type in EventType.LIVE_ONLY:
            return
        self._ensure_head()
        event.seq = self._seq
        self._seq += 1
        try:
            line = dumps(event)
        except TypeError as exc:             # E4：点明是谁，而不是裸抛
            raise SessionError(
                f"事件 {event.type!r} 的 data 里有无法 JSON 序列化的值: {exc}"
            ) from exc
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        self.messages = _apply_to_window(self.messages, event)

    def write_events(self, events: Iterable[Event]) -> None:
        """写一串事件。**会完整消费**这个迭代器。"""
        self._ensure_head()

        with open(self.path, "a", encoding="utf-8") as handle:

            for event in events:
                if event.type in EventType.LIVE_ONLY:
                    continue

                event.seq = self._seq
                self._seq += 1

                try:
                    line = dumps(event)
                except TypeError as exc:  # E4：点明是谁，而不是裸抛
                    raise SessionError(
                        f"事件 {event.type!r} 的 data 里有无法 JSON 序列化的值: {exc}"
                    ) from exc

                handle.write(line + "\n")
                self.messages = _apply_to_window(self.messages, event)



    # ---- 读（重放） ----
    def events(self) -> list[Event]:
        """按文件顺序读回所有事件。空行与崩溃造成的半行被跳过（前缀仍合法）。"""
        return _read_events(self.path)

    def replay(self) -> Messages:
        """重放出**裁剪过**的历史 —— 恢复点保证是合法消息边界。"""
        return replay_messages(self.events())

def _read_events(path: str) -> list[Event]:
    out: list[Event] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = loads(line)
            except json.JSONDecodeError:
                continue                       # 最后一行可能因崩溃而残缺
            if isinstance(record, Event):
                out.append(record)
    return out


def _read_head(path: str) -> SessionHead:
    with open(path, "r", encoding="utf-8") as handle:
        first = handle.readline().strip()
    record = loads(first) if first else None
    if not isinstance(record, SessionHead):
        raise SessionError(f"会话文件头缺失或损坏: {path}")
    return record

class SessionStore:
    """会话仓库：一个目录，每个会话一个 ``.jsonl`` 文件。"""

    def __init__(self, root: str = ".harness/sessions") -> None:
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, session_id: str) -> str:
        return os.path.join(self.root, f"{session_id}.jsonl")

    # ---- 读文件头（做映射用；只读第一行，不加载事件） ----
    def iter_heads(self, *, parent: int | str = -1) -> list[tuple[SessionHead, str]]:
        """逐个读文件头，返回 ``[(head, path)]``；**跳过损坏的文件**。

        为什么跳过而不是报错：列表 / 映射是"尽力而为"的操作，一个坏文件不该让整个
        仓库不可用（坏文件仍会被 ``open(session_id)`` 明确报出来 —— 那里 ``_read_head`` 照样抛）。
        """
        found: list[tuple[SessionHead, str]] = []
        for name in sorted(os.listdir(self.root)):
            if not name.endswith(".jsonl"):
                continue
            path = os.path.join(self.root, name)
            try:
                head = _read_head(path)
            except (SessionError, OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if parent is None or head.parent == parent:
                found.append((head, path))
        return found


    def child_ids(self, parent: int | str) -> list[str]:
        """某个会话的**附属会话** id（子代理写的 ``<parent>.sub-N``），按 id 排序。

        这是 ``iter_heads(parent=...)`` 那套判据的公开形状。子代理层要用它回答"这个父会话
        下面已经有哪些编号被占了" —— 编号不能凭空从 1 数起（见 `src/harness/subagent.py`
        的 ``_next_child_id``：内存计数器在重新装配之后会从 1 重来，而 ``.sub-1`` 还在磁盘上）。
        """
        return sorted(head.session_id for head, _ in self.iter_heads(parent=parent))

    def events_of(self, session_id: str) -> list[Event]:
        """**只读**地读某条会话的全部事件（不建上下文窗口、不改任何状态）。

        给"看一条**不属于当前 harness** 的会话"用：子会话视图要的是事件流，
        而不是重放出来的消息窗口。`open()` 会顺手做后者（还得多读一遍文件），
        而它唯一的用途是**接着跑** —— 看，不该有那个副作用。
        """
        path = self._path(session_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"会话不存在: {session_id}")
        return _read_events(path)

    def find_main(self, ws_key: str) -> list[tuple[SessionHead, str]]:
        """某个工作区的**主控**会话，按"最近活动"倒序（新 → 旧）。

        用文件 ``mtime`` 排序（不是会话 id —— id 里的时间戳只是创建时刻）。
        扫 ``head.meta.workspace`` 而不是维护一张索引表：索引表在迁移 / 改名 / 手工拷文件之后
        必然和实际文件不一致，而不一致**不会有任何报错**，只会表现成"接不上上次的会话"。

        工作区被移动 / 改名后，键就变了 —— 表现为"这是个新工作区"，旧会话成为孤儿
        （**不丢、不覆盖**：老文件仍指向老路径）。
        """
        matched = [(head, path) for head, path in self.iter_heads()
                   if head.meta.get("workspace") == ws_key]
        matched.sort(key=lambda item: os.path.getmtime(item[1]), reverse=True)
        return matched

    def create(self, session_id: str = None, *, parent: int | str = -1, meta: dict[str, Any] | None = None) -> Session:
        """新建会话：写文件头，**上下文窗口为空**。"""
        sid = session_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        path = self._path(sid)
        if os.path.exists(path):
            raise FileExistsError(f"会话已存在，拒绝覆盖: {sid}（{path}）")
        session = Session(path, sid, parent, meta)
        session._ensure_head()
        return session

    def open(self, session_id: str) -> Session:
        """重新打开会话：读文件头 + 重放出**最新的上下文窗口**。"""
        path = self._path(session_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"会话不存在: {session_id}")
        head = _read_head(path)
        session = Session(path, head.session_id, head.parent, head.meta)
        session._has_head = True
        events = session.events()
        session.messages = replay_messages(events)
        session._seq = max((event.seq for event in events), default=-1) + 1   # seq 接着编号
        return session

    def load_ui_events(self, session_id: str) -> list[Event]:
        """加载"用户界面可见内容"的事件（**人类 / AI / 工具**三类，按文件顺序）。

        只给三类**进历史**的内容事件，过程事件（``run_start`` / ``model_start`` /
        ``tool_start`` …）不在其中。所以它适合"只要对话内容"的消费者（回放一句话摘要、
        导出对话）。

        ⚠ **要工具参数就别用它**：参数只在 ``tool_start`` 上（不在 ``HISTORY`` 里），
        从这里读不到。界面时间线现在走的是全量事件（:meth:`Session.events`）＋ 按 ``span``
        把 ``tool_start`` 与 ``tool_result`` 配对 —— 少了前者，工具那一行只能写成
        ``read · ?``，而参数本来就在文件里。
        """
        path = self._path(session_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"会话不存在: {session_id}")
        return [event for event in _read_events(path) if event.type in EventType.HISTORY]

    def list(self) -> list[str]:
        return sorted(head.session_id for head, _ in self.iter_heads())

# ---------------------------------------------------------------------------
# 存储
# ---------------------------------------------------------------------------
def workspace_key(path: str) -> str:
    """工作区路径 → 用于映射的规范键。

    会话文件存在**程序侧**，靠"工作区 → 主控会话文件"的映射定位。这个映射的正确性
    全在键怎么算：同一个工作区算出两个键，就会**静默生成两份轨迹**（不报错、只是接不上
    上次的会话），所以三种写法必须归一：

    - ``realpath``：软链接、``..``、Windows 的 8.3 短名 —— ``abspath`` 只做字符串拼接，
      会给同一个目录算出不同的字符串；
    - ``normcase``：Windows 上 ``D:\\a`` 与 ``d:\\A`` 是同一个目录（Linux 上此函数不做改动，
      所以不需要写平台分支）；
    - ``normpath``：去掉结尾分隔符（``D:\\a`` 与 ``D:\\a\\`` 同样是两个字符串）。

    键本身**进 ``head.meta.workspace``**：评估要按工作区分组，记用户传进来的原始字符串
    会把同一个项目分成两组。
    """
    return os.path.normcase(os.path.normpath(os.path.realpath(os.path.abspath(path))))