import json
import time
import uuid
from dataclasses import field, dataclass
from enum import StrEnum, nonmember
from typing import Any

from src.core.message import from_jsonable, to_jsonable


class EventType(StrEnum):
    """事件类型"""

    # ---- 会话与 run ----
    SESSION_START = "session_start"  # 时间线起点（纯标记）；同时开会话 span（树的最外层根）
    SESSION_END = "session_end"  # 时间线终点；关会话 span
    RUN_START = "run_start"  # 一次 run（= 用户一次请求）开始
    RUN_END = "run_end"  # 关 run span；带 stop_reason / usage
    SYS_PROMPT = "sys_prompt"  # 模型可见的静态上下文（prompt ＋ 工具清单 ＋ 记忆来源）
    USER_MESSAGE = "user_message"  # 进历史

    # ---- 模型 ----
    MODEL_START = "model_start"  # 发起一次模型调用（唯一能测"卡在模型上"的开口）
    ASSISTANT_MESSAGE = "assistant_message"  # 进历史；usage / stop_reason 在消息里
    # 同时**关掉模型 span**，并带 duration_ms / ttft_ms
    TEXT_DELTA = "text_delta"  # 打字机增量：**只实时推送，不落盘**

    REASONING = "reasoning"

    # ---- 工具与委派 ----
    TOOL_START = "tool_start"  # data.tool_type 区分普通工具 / 子代理委派
    TOOL_RESULT = "tool_result"  # 关 span ＋ 进历史

    # ---- 上下文与异常 ----
    COMPACTION = "compaction"  # 改写历史（摘要 / 清理旧工具结果）
    ERROR = "error"

    # ---- 人机交互（**只实时，不落盘**） ----
    APPROVAL_REQUEST = "approval_request"

    # ---- 开事件集合 ---
    OPENING = nonmember(frozenset({SESSION_START, RUN_START, MODEL_START, TOOL_START}))

    # ---- 关事件集合 ---
    CLOSING = nonmember(frozenset({SESSION_END, RUN_END, TOOL_RESULT, ASSISTANT_MESSAGE}))

    # ---- 历史事件集合 ---
    HISTORY = nonmember(frozenset({USER_MESSAGE, ASSISTANT_MESSAGE, TOOL_RESULT}))

    # ---- 流事件 ---
    LIVE_ONLY = nonmember(frozenset({TEXT_DELTA, REASONING, APPROVAL_REQUEST}))


class ToolType(StrEnum):
    """工具类型"""

    TOOL = "tool"
    SUBAGENT = "subagent"


class Outcome(StrEnum):
    """操作怎么结束"""

    OK = "ok"
    FAILED = "failed"
    DENIED = "denied_by_policy"  # 被权限策略拒绝（≠ 工具本身出错）
    BLOCKED = "blocked_by_hook"  # 被钩子拦截
    TIMEOUT = "timeout"  # 超时
    CANCELLED = "cancelled"  # 上层取消（用户按 Esc / 关窗口）
    INTERRUPTED = "interrupted"  # 被 ``interrupt()`` 打断


class CompactionStrategy(StrEnum):
    """上下文压缩策略"""

    CLEAR_TOOL_RESULTS = "clear_tool_results"
    SUMMARIZE = "summarize"


class StopReason(StrEnum):
    """Agent loop 执行一次停止的原因"""

    END_TURN = "end_turn"  # 正常 loop 结束
    MAX_TURNS = "max_turns"  # 到达单轮 loop 限制的最大论次数
    INTERRUPTED = "interrupted"  # interrupt() 生效
    BUDGET_EXCEEDED = "budget_exceeded"  # 超硬预算、且两档压缩都压不动
    STALLED = "stalled"  #: 无进展（连续相同回复 / 工具风暴）被 loop 主动判停 —— 不是撞上 max_turns，是"卡死了"
    ERROR = "error"
    CANCELLED = "cancelled"


class ErrorReason(StrEnum):
    """异常原因"""

    PROVIDER = "provider"
    TOOL = "tool"
    PERMISSION = "permission"
    BUDGET = "budget"
    CANCEL = "cancel"
    INTERNAL = "internal"


class DataKey(StrEnum):
    """事件的 ``data`` 里被多方引用的键名"""

    DURATION_MS = "duration_ms"  # 这个动作/这段操作花了多久
    OUTCOME = "outcome"  # 取值见 ``Outcome``（关 span 的事件都有）
    MESSAGE = "message"  # 进历史的那条消息
    TRUNCATED_FROM = "truncated_from"  # 工具结果被截断时，记**原始长度**
    SPILLED_TO = "spilled_to"  # 被截断的全文落在哪个文件（相对工作区，机器可读）
    TTFT_MS = "ttft_ms"  # 首字延迟：**只有这里能记**（text_delta 不落盘）
    CHUNK_COUNT = "chunk_count"  # 流式增量条数（模型吐字的"节奏"）

    TOOL_TYPE = "tool_type"  # 这次工具调用调的是什么，取值见 ``ToolType``
    CALL_ID = "call_id"  # 工具调用的 id（tool_start 里与 message.tool_call_id 对应）
    CHILD_SESSION = "child_session"  # 子代理的会话 id → 它的附属文件（看图往下钻）
    USAGE = "usage"  # 用量（在产生它的那条记录上）
    STOP_REASON = "stop_reason"  # run 怎么结束的，取值见 ``StopReason``
    ERROR_SOURCE = "error_source"  # 出错来源


class MetaKey(StrEnum):
    """``head.meta`` 的约定键。``meta`` 是可扩展袋，**不认识的键忽略**（加东西不用升 ``v``）。"""

    AGENT = "agent"
    WORKSPACE = "workspace"  # 跑在哪个仓库 —— 事后从事件里**再也拿不到**
    MODEL = "model"
    PROVIDER = "provider"
    #: harness 的**代码**版本（建议带 git sha，如 ``0.2.0+git.a1b2c3d``）。注意它和文件头
    #: ``v``（**数据格式**版本）是两件事：跨版本对比轨迹时，靠它判断"这次是哪个 harness 跑的"。
    HARNESS = "harness"
    #: 配置指纹：``parallel_tools`` / ``max_turns`` / ``temperature`` / ``permission_mode`` …
    #: 这些直接决定行为。**``parallel_tools`` 尤其关键** —— 并行与串行的轨迹形状不同，
    #: 不记下来两份轨迹根本不可比。
    CONFIG = "config"

    #: ---- 下面三个是**评估注入**的，不进 ``REQUIRED``（跑真任务时才有） ----
    #: 被试任务的标识。评估按它分组算 pass@k；缺了就只能按文件分，一组一次尝试。
    TASK_ID = "task_id"
    #: 同一个任务的第几次尝试（从 1 起）。同一个 task_id 可以有很多次尝试。
    ATTEMPT = "attempt"
    #: 随机种子（如果这次跑用到了随机性）。复现一次可疑的失败轨迹要靠它。
    SEED = "seed"

    #: 写文件的一方必须填齐这些键；评估靠它们做跨版本 / 跨配置对比。
    REQUIRED = nonmember(frozenset({AGENT, WORKSPACE, MODEL, PROVIDER, HARNESS, CONFIG}))


class ApprovalSource(StrEnum):
    """**这个决定是谁做的**

    这个字段不是装饰：评估要把"人真的审了"和"人不在、按默认值放行/拒绝"分开 ——
    两者的意义完全相反（前者衡量 harness 的自主性，后者衡量配置有多危险）。
    """

    #: 人点的（弹窗上按的批准 / 拒绝）。
    HUMAN = "human"
    #: 弹窗自己的倒计时到点 → 按风险等级的默认值裁决。**人在场但没决定。**
    TIMEOUT = "timeout"
    #: 闸门层的硬兜底超时 → 默认值裁决。**说明审批通道（TUI）自己出事了**（该报警）。
    TIMEOUT_BACKSTOP = "timeout_backstop"
    #: 没有审批通道（没给 channel / approver）→ 直接拒绝。**当前 fail-safe 的默认路径。**
    NONE = "none"


class ApprovalDecision(StrEnum):
    """审批结果"""

    APPROVED = "approved"
    DENIED = "denied"


@dataclass(frozen=True)
class ApprovalRequest:
    """一次待审批的调用 —— 消费者（TUI / CLI / 测试）拿它渲染弹窗、并 ``send`` 回决定。

    它是**自足**的：``call`` / ``invocation`` 说明"要批什么"，``reason`` / ``risk`` 说明
    "为什么问、有多危险"，``timeout_s`` 说明"最多等多久"，``asked_ts`` / ``expires_ts``
    让 UI 能画倒计时（与 :func:`now_ms` 同一时钟）。
    """

    call: Any  # ToolCall（避免与 messages 互相 import）
    reason: str  # 策略给的理由（为什么问）
    risk: str  # read / write / exec / spawn
    invocation: str  # 一行摘要，给弹窗当标题
    asked_ts: int  # 墙上时间（毫秒），与 now_ms() 同源
    timeout_s: float  # 弹窗自己该等多久
    expires_ts: int  # asked_ts + timeout_s×1000，画倒计时用
    proposed: bool = False  # 超时 / 无人回答时会被当成什么（给人看后果）


def now_ms() -> int:
    """获取整数毫秒时间戳"""
    return int(time.time() * 1000)


def get_new_id() -> str:
    """产生一个 12 位长度的 id"""
    return uuid.uuid4().hex[:12]


@dataclass
class SessionHead:
    """会话头"""

    session_id: str
    parent: int | str = -1  #: 负数 = 主控（没有父）；否则是**父会话的 ``session_id``**。
    meta: dict[str, Any] = field(default_factory=dict)
    v: int = 1

    @property
    def is_main(self) -> bool:
        """是不是主控"""
        return isinstance(self.parent, int) and not isinstance(self.parent, bool) and self.parent < 0

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "head",
            "v": self.v,
            "session_id": self.session_id,
            "parent": self.parent,
            "meta": self.meta,
        }

    def missing_meta(self) -> frozenset:
        """还缺哪些**评估必需**的 meta 键。

        写文件的一方必须保证结果为空（评估要按 ``harness`` / ``config`` 做跨版本对比，
        缺了就说不清"这次的差异是代码带来的还是配置带来的"）。
        """
        return MetaKey.REQUIRED - set(self.meta)


# ---------------------------------------------------------------------------
# 事件
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """一条事件

    - ``seq`` 由会话层在写文件时按**文件内自增**赋值（这里默认 0）。
    - ``ts`` 默认取当前毫秒。
    - ``span`` **每条事件都有**（"我属于哪个操作"）；``parent_span`` **只在开 span 的事件上**
      （"我这个操作套在谁里面"）。
    """

    type: str
    data: dict[str, Any] = field(default_factory=dict)
    seq: int = 0
    ts: int = field(default_factory=now_ms)
    run_id: str | None = None
    turn: int | None = None
    span: str | None = None
    parent_span: str | None = None

    def __post_init__(self) -> None:
        # ⚠SP：``parent_span`` 只在开 span 的事件上。这里做个便宜的护栏，
        # 把"不小心在 model_end / tool_result 上也写了 parent_span"当场拦下来。
        #
        # 只在**认识这个 type** 时校验：不认识的类型必须能构造出来，

        if (
                self.parent_span is not None
                and self.type in EventType  # ← 认识的类型
                and self.type not in EventType.OPENING
        ):
            raise ValueError(
                f"{self.type!r} 不是开 span 的事件，不应该带 parent_span（契约 ⚠SP）；"
                f"开 span 的事件是 {sorted(EventType.OPENING)}"
            )

    def to_jsonable(self) -> dict[str, Any]:
        """转成一行记录的内容。值为 ``None`` 的可选字段**不落盘**（省体积，读回来仍是 ``None``）。"""
        out: dict[str, Any] = {"seq": self.seq, "ts": self.ts}
        for key in ("run_id", "turn", "span", "parent_span"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        out["type"] = self.type
        out["data"] = to_jsonable(self.data)
        return out


def event_from_jsonable(obj: dict[str, Any]) -> Event:
    return Event(
        type=obj["type"],
        data=from_jsonable(obj.get("data") or {}),
        seq=obj.get("seq", 0),
        ts=obj.get("ts", 0),
        run_id=obj.get("run_id"),
        turn=obj.get("turn"),
        span=obj.get("span"),
        parent_span=obj.get("parent_span"),
    )

def head_from_jsonable(obj: dict[str, Any]) -> SessionHead:
    return SessionHead(
        session_id=obj["session_id"],
        parent=obj.get("parent", -1),
        meta=from_jsonable(obj.get("meta") or {}),
        v=obj.get("v", 1),
    )
# ---------------------------------------------------------------------------
# 一行 JSON ↔ 一条记录
# ---------------------------------------------------------------------------

def dumps(record: Event | SessionHead) -> str:
    """一条记录 → 一行 JSON。

    契约第 2 节：**``ensure_ascii=False``**（中文按 UTF-8 原样写；否则每个汉字变成
    ``\\uXXXX`` 六个字节，体积翻倍）。行尾换行由写文件的一方补。

    两种记录的序列化方法名历史上不一致（``Event.to_jsonable`` / ``SessionHead.to_json``），
    这里统一掉 —— 否则按 ``record.to_jsonable()`` 写死，写文件头时会 ``AttributeError``。
    """
    to_mapping = (record.to_jsonable if hasattr(record, "to_jsonable") else record.to_json)
    return json.dumps(to_mapping(), ensure_ascii=False)


def loads(line: str) -> Event | SessionHead:
    """一行 JSON → 记录。``type == "head"`` 的是文件头，其余是事件。

    分派键是**字符串** ``"head"``（``SessionHead.to_json()`` 写出去的那个值），
    不是 ``SessionHead`` 这个**类对象** —— 拿类去比字符串永远为假，结果是文件头
    被当成事件解析（不报错，静默还原成错类型）。
    """
    obj = json.loads(line)
    if obj.get("type") == "head":
        return head_from_jsonable(obj)
    return event_from_jsonable(obj)