import dataclasses
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import field, dataclass
from typing import Any, Union




@dataclass
class BaseMessage(ABC):
    content: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    @abstractmethod
    def role(self) -> str:
        pass


@dataclass
class SystemMessage(BaseMessage):

    @property
    def role(self) -> str:
        return "system"

@dataclass
class HumanMessage(BaseMessage):
    @property
    def role(self) -> str:
        return "user"

@dataclass
class ToolMessage(BaseMessage):

    tool_call_id: str = field(default=...)
    name: str = field(default=...)
    is_error: bool = field(default=False)

    @property
    def role(self) -> str:
        return "tool"


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0

    def __post_init__(self) -> None:
        self.input_tokens = self.input_tokens or 0
        self.output_tokens = self.output_tokens or 0
        self.reasoning_tokens = self.reasoning_tokens or 0
        self.cached_tokens = self.cached_tokens or 0
        self.total_tokens = self.input_tokens + self.output_tokens

    def __add__(self, other: "Usage") -> "Usage":
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )

@dataclass
class ToolCall:
    id: str = field(default=...)
    name: str = field(default=...)
    args: dict[str, Any] = field(default_factory=dict)

@dataclass
class AIMessage(BaseMessage):

    usage: Usage  = field(default_factory=Usage)
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = field(default=None)
    reasoning_content: str | None = field(default=None)

    @property
    def role(self) -> str:
        return "assistant"

    def tool_calls_to_json(self):
        if not self.tool_calls:
            return []
        calls = []
        for tool_call in self.tool_calls:
            calls.append({
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.name,
                "arguments" : json.dumps(tool_call.args)
            }
        })
        return calls

AnyMessage = Union[SystemMessage, HumanMessage, ToolMessage, AIMessage]
Messages = list[AnyMessage]

# ---------------------------------------------------------------------------
# 编解码：**对称**，落盘再读回来不丢类型
# ---------------------------------------------------------------------------

_TYPE_KEY = "__type__"

_CODEC_TYPES: dict[str, type] = {
    cls.__name__: cls
    for cls in (SystemMessage, HumanMessage, AIMessage, ToolMessage, ToolCall, Usage)
}


def to_jsonable(obj: Any) -> Any:
    """把消息 / ``ToolCall`` / ``Usage`` / 容器转成可 JSON 化的结构。

    每个需要还原的对象都带一个 ``__type__`` 标记。

    """
    if isinstance(obj, (BaseMessage, ToolCall, Usage)):
        out: dict[str, Any] = {_TYPE_KEY: type(obj).__name__}
        for f in dataclasses.fields(obj):
            out[f.name] = to_jsonable(getattr(obj, f.name))
        return out
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def from_jsonable(obj: Any) -> Any:
    """``to_jsonable`` 的逆操作。

    两条向前兼容规则：

    - **不认识的 ``__type__`` 原样返回**（不抛异常）—— 新版本写的消息，老版本读到
      只当普通 dict，而不是崩掉；
    - 认识类型时，**多余的键被忽略**（新版本给消息加字段时，老版本仍能读）。
    """
    if isinstance(obj, list):
        return [from_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        cls = _CODEC_TYPES.get(obj.get(_TYPE_KEY))
        if cls is not None:
            allowed = {f.name for f in dataclasses.fields(cls)}
            kwargs = {
                k: from_jsonable(v)
                for k, v in obj.items()
                if k != _TYPE_KEY and k in allowed
            }
            return cls(**kwargs)
        return {k: from_jsonable(v) for k, v in obj.items()}
    return obj
