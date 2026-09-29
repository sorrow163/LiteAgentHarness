import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import replace
from typing import Any, Sequence

from src.core.events import EventType
from src.core.message import Messages, AIMessage, Usage, ToolCall
from src.core.tool import Tool

_REASONING_ECHO_MODELS = ("deepseek-v4-flash", "deepseek-v4-pro")
def needs_reasoning_echo(model: str) -> bool:
    """这个模型是否属于"必须回传推理内容"的那一类。

    按名字前缀匹配（``deepseek-v4-flash`` 与 ``deepseek-v4-flash-20260101`` 都算）。
    """
    name = (model or "").lower()
    return any(name.startswith(known) for known in _REASONING_ECHO_MODELS)

def dpsk_thinking_disabled(kwargs: dict[str, Any]) -> bool:
    """调用参数里是否**明确**关掉了思考模式。

    只看参数形状，**不看 provider 名**：``extra_body`` 里写了 ``thinking``（``"disabled"``、
    或 ``{"type": "disabled"}``）就是关了。以前要先判 ``provider == "deepseek"`` —— 那等于把
    "哪家端点认这个形状"硬编码成一张只会过期的表；而这里本来就是**白名单式**的保守判断
    （认不出来就当没关），所以去掉 provider 只会更准，不会更激进。
    """
    extra = kwargs.get("extra_body")
    if not isinstance(extra, dict):
        return False
    thinking = extra.get("thinking", extra)
    if isinstance(thinking, str):
        return thinking.lower() == "disabled"
    if isinstance(thinking, dict):
        return str(thinking.get("type", "")).lower() == "disabled"
    return False

_THINKING_DISABLED = {"deepseek": dpsk_thinking_disabled }

class BaseChatModel(ABC):
    """模型接口。``bind_tools`` 把工具挂在模型上（``invoke`` 时一并发出去）。"""

    def __init__(self) -> None:
        self._tools: list[Tool] = []
        self.tools_schema = []


    def bind_tools(self, tools: Sequence[Tool]) -> "BaseChatModel":
        self._tools = list(tools)
        self.tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.bound_tools
        ]
        return self

    @property
    def bound_tools(self) -> list[Tool]:
        return list(getattr(self, "_tools", []))

    @abstractmethod
    def invoke(self, messages: Messages) -> AIMessage:
        raise NotImplementedError

    def stream(self, messages: Messages) -> Iterator[str | AIMessage]:
        """默认实现：不支持流式的模型退化为"一次性吐全文"。

        这保证**任何**模型（包括 ``FakeModel``）都能被流式消费者驱动 ——
        消费者的写法永远是"循环收增量，最后一项当结果"。
        """
        ai = self.invoke(messages)
        if ai.content:
            yield ai.content
        yield ai


class FakeModel(BaseChatModel):
    """离线跑测试的假模型，按剧本依次返回响应"""

    def __init__(self, script: Sequence[Any] = ()) -> None:
        super().__init__()
        self.script = list(script)
        self.calls: list[Messages] = []      # 每轮实际收到的消息（测试断言用）

    def invoke(self, messages: Messages) -> AIMessage:
        self.calls.append(list(messages))
        item = self.script.pop(0) if self.script else "(剧本已结束)"
        if callable(item):
            item = item(messages)

        if isinstance(item, AIMessage):
            ai = replace(item, tool_calls=list(item.tool_calls), usage=item.usage)
        elif isinstance(item, list):
            ai = AIMessage(content="", tool_calls=list(item))
        else:
            ai = AIMessage(content=str(item))

        if not ai.stop_reason:
            ai.stop_reason = "tool_use" if ai.tool_calls else "end_turn"

        if not ai.usage.total_tokens:
            ai.usage = Usage(
                input_tokens=sum(len(m.content or "") for m in messages) // 4,
                output_tokens=len(ai.content or "") // 4,
                cached_tokens= 0)
        return ai

def _usage_from(raw: Any) -> Usage:
    """provider 的 usage → 实例化 ``Usage``"""
    if raw is None:
        return Usage()
    completion_details = getattr(raw, "completion_tokens_details", None)
    prompt_details = getattr(raw, "prompt_tokens_details", None)
    return Usage(
        input_tokens=getattr(raw, "prompt_tokens", 0) or 0,
        output_tokens=getattr(raw, "completion_tokens", 0) or 0,
        reasoning_tokens=getattr(completion_details, "reasoning_tokens", 0) or 0,
        cached_tokens=getattr(prompt_details, "cached_tokens", 0) or 0,
    )


def _parse_args(raw: str) -> dict[str, Any]:
    """工具参数的 JSON 字符串 → dict。

    模型偶尔会吐出坏 JSON（流式分片拼起来时尤其容易）。老实现直接 ``json.loads`` ——
    一个坏参数就把整个 run 炸掉。这里退化成空 dict：工具的参数校验随后会报
    "缺少这个参数"，模型看得见、能自己重试。
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}

def _to_openai(messages: Messages, need_reasoning_content: bool = False) -> list[dict[str, Any]]:
    """Messages → OpenAI 兼容的 messages"""

    out: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, AIMessage):
            # 带工具调用时可以给 null（provider 都接受）；纯文本消息给空串，别给 null。
            content = message.content or (None if message.tool_calls else "")
            item: dict[str, Any] = {"role": "assistant", "content": content}
            if message.tool_calls:
                item["tool_calls"] = message.tool_calls_to_json()
            if need_reasoning_content and message.reasoning_content:
                item["reasoning_content"] = message.reasoning_content
            out.append(item)
        elif message.role == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            })
        else:
            out.append({"role": message.role, "content": message.content})
    return out

class OpenAIChatModel(BaseChatModel):
    """任何 OpenAI 兼容服务（``pip install openai``），换 ``base_url`` 即换 provider。

    ★ **``provider`` 不是必需参数**（默认 ``"unknown"``）：它不影响请求怎么发，只用在两处
    —— 写进会话头的元信息（评估要按它分组/对比），以及"哪家的思考模式写法"这类判断。
    所以它降级成一个**可选的标注**，而不是构造模型的先决条件。
    """

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 provider: str = "unknown", **kwargs: Any):
        super().__init__()
        from openai import OpenAI          # 延迟导入：没装 openai 也能用 FakeModel 跑测试

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.provider = provider
        self.model = model
        self.kwargs = kwargs
        #: 要不要把推理内容回传 —— **两个条件都要满足**：
        #: ① 这个模型属于已知需要回传的那一类（白名单）；② 没有明确把思考模式关掉。
        thinking_disabled_func = _THINKING_DISABLED.get(provider, None)
        thinking_disabled = False if not thinking_disabled_func else thinking_disabled_func(kwargs)
        self.need_reasoning_content = needs_reasoning_echo(model) and not thinking_disabled

    # ---- 请求组装 ----
    def _params(self, messages: Messages) -> dict[str, Any]:

        params: dict[str, Any] = dict(
            model=self.model,
            messages=_to_openai(messages, self.need_reasoning_content),
            **self.kwargs,
        )
        if self.tools_schema:
            params["tools"] = self.tools_schema
        return params

    # ---- 非流式 ----
    def invoke(self, messages: Messages) -> AIMessage:
        response = self.client.chat.completions.create(**self._params(messages))
        choice = response.choices[0]
        message = choice.message
        tool_calls = [
            ToolCall(name=call.function.name,
                     args=_parse_args(call.function.arguments),
                     id=call.id)
            for call in (message.tool_calls or [])
        ]
        return AIMessage(
            content=message.content or "",
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason or "",
            usage=_usage_from(getattr(response, "usage", None)),
            # thinking 模型（DeepSeek 等）把思考内容单独一路返回；不回填的话，
            # _to_openai 里的回传通道就是空转的。
            reasoning_content=getattr(message, "reasoning_content", "") or "",
        )

    # ---- 流式 ----
    def stream(self, messages: Messages) -> Iterator[dict | AIMessage]:
        # OpenAI 的流式要自己累积：文本按增量拼接，工具调用的参数是
        # **分片送达的 JSON 字符串**，必须按 index 归位、最后再整体解析。
        params = self._params(messages)
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}

        text, finish, reasoning = "", "", ""
        usage = Usage()
        calls: dict[int, dict[str, str]] = {}
        for chunk in self.client.chat.completions.create(**params):
            if getattr(chunk, "usage", None):
                usage = _usage_from(chunk.usage)
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish = choice.finish_reason
            delta = choice.delta
            if delta is None:
                continue
            if delta.content:
                text += delta.content
                yield {EventType.TEXT_DELTA: delta.content}
            # reasoning_content 是**单独一路**增量：不能混进 content 当正文吐给上层，
            # 只累积进最终消息，供 _to_openai 回传使用。
            #
            # ⚠ **形状契约：键就是事件类型**（和上面 TEXT_DELTA 那一行同一套），
            # 因为循环是按 `item[EventType.X]` 取的（见 `loop._run` 收增量那一段）。
            # 这里曾经写成 `{"type": ..., "content": ...}` —— 键对不上，于是**思考内容
            # 一路都不产生 `reasoning` 事件**：模型明明在想，轨迹里、界面上一个字都没有，
            # 而且**全链路不报错**（`in` 判断为假就静默跳过）。由
            # `tests/test_stream_events.py` 盯着这条链路。
            reasoning_delta = getattr(delta, "reasoning_content", None)
            if reasoning_delta:
                reasoning += reasoning_delta
                yield {EventType.REASONING: reasoning_delta}
            for call in (delta.tool_calls or []):
                slot = calls.setdefault(call.index, {"id": "", "name": "", "args": ""})
                if call.id:
                    slot["id"] = call.id
                if call.function and call.function.name:
                    slot["name"] += call.function.name
                if call.function and call.function.arguments:
                    slot["args"] += call.function.arguments

        tool_calls = [
            ToolCall(name=slot["name"],
                     args=_parse_args(slot["args"]),
                     id=slot["id"] or f"call_{index}")
            for index, slot in sorted(calls.items())
        ]
        yield AIMessage(content=text, tool_calls=tool_calls, stop_reason=finish,
                        usage=usage, reasoning_content=reasoning)


