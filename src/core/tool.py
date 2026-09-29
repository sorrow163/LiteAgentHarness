import inspect
import json
import os
import re
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, get_type_hints, Annotated

from pydantic import create_model, ConfigDict, BaseModel, Field, ValidationError

from src.core.errors import ToolBindError
from src.core.events import ToolType, Outcome
from src.core.message import ToolCall, ToolMessage, Usage
from src.core.runtime import set_call, CallContext, reset_call

#: 工具名的规则。模型就是靠这个名字发起调用的，所以不许有奇怪字符。
NAME_PATTERN = re.compile(r"^[a-z_][a-z0-9_]*$")

# 工具函数参数使用 pydantic 模型的最大深度限制
MAX_REF_DEPTH = 12

#: 结果落盘目录（相对 workspace）。放在 .harness/ 下 —— 建议加进 .gitignore。
DEFAULT_RESULT_DIR = ".harness/results"

#: 单个结果文件的写盘上限：超过就只写前 2 MB。
MAX_RESULT_FILE_BYTES = 2 * 1024 * 1024


def _clean_json_schema(part: Any, refs: Mapping[str, Any], depth: int) -> Any:
    """整理 schema 里的**一块**（part ＝ JSON 树上的一个节点）。

    JSON Schema 本身就是一棵树：dict 和 list 是"枝"，字符串 / 数字 / 布尔 / null 是"叶"。
    这个函数**每次只管手上这一块**：

    - 这块是**列表**（比如 ``anyOf`` / ``required``）→ 每个元素各处理一遍
    - 这块是**叶** → 原样返回，递归就停在这里
    - 这块是**对象** → 有 ``$ref`` 就先换成真身；没有 ``$ref`` 就逐个键处理，
      顺手丢掉 ``title`` 与 ``$defs``

    之所以要走遍整棵树：``title`` 被 pydantic 放在**每一层**的每个属性上，
    ``$ref`` 也可能出现在**任意一层**（嵌套的 BaseModel 参数被塞进了 ``$defs``，
    参数位置只留一个 ``$ref``），只处理顶层是不够的。

    ``depth`` 只在**跨过一次 ``$ref``** 时 +1，所以它数的是"引用套引用的层数"：
    普通嵌套（对象里套对象）不会把它推高，只有定义自我引用才会 —— 那种情况当场报错，
    总比把一个还带 ``$ref`` 的 schema 交给 provider 去猜要好。
    """
    # 枝之一：列表
    if isinstance(part, list):
        cleaned_items = []
        for item in part:
            cleaned_items.append(_clean_json_schema(item, refs, depth))
        return cleaned_items

    # 叶：字符串 / 数字 / 布尔 / null → 原样返回
    if not isinstance(part, dict):
        return part

    # 枝之二：对象。先看它是不是 $ref（指向 $defs 里的"真身"）
    if "$ref" in part:
        if depth > MAX_REF_DEPTH:
            raise ToolBindError(
                f"{part['$ref']!r} 的定义在自我引用（引用嵌套超过 {MAX_REF_DEPTH} 层），"
                "本内核不支持这种参数模型"
            )
        ref_name = str(part["$ref"]).rsplit("/", 1)[-1]
        if ref_name not in refs:
            raise ToolBindError(f"找不到 {ref_name!r} 的 json schema")
        # 真身是一棵完整的 schema 树，交给同一个函数递归处理
        merged = _clean_json_schema(refs[ref_name], refs, depth + 1)
        # $ref 旁边挂着的键（如 description）覆盖真身里的同名键
        for key, value in part.items():
            if key != "$ref":
                merged[key] = _clean_json_schema(value, refs, depth)
        return merged

    # 普通对象：逐个键处理（顺手丢掉 title 与 $defs）
    cleaned: dict[str, Any] = {}
    for key, value in part.items():
        if key in ("title", "$defs", "definitions"):
            continue
        # 注意：value 可能是 dict，也可能是**列表**（required / anyOf / enum），
        # 甚至可能是叶子。下一层的 part 就是它 —— 所以函数开头才要判断 list。
        cleaned[key] = _clean_json_schema(value, refs, depth)
    return cleaned


def normalize_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """把 pydantic 生成的 JSON Schema 归一化成 provider 友好的形状。

    走一遍整棵树，顺手做三件事：

    ① 遇到 ``$ref`` → 把 ``$defs`` 里的真身**就地展开**。``$ref`` ＋ ``$defs`` 是
       schema 复用的机制，不是给模型看的东西：模型看到的 ``params`` 若是一句
       ``{"$ref": "#/$defs/ToolParam"}``，它得先学会"顺着 $defs 找定义"这条我们
       没写在提示里的约定。展开成字面量，参数长什么样就摆在那儿。
    ② 丢掉 ``title``和 ``$defs`` 本体

    ③ 保证顶层是 ``type: object``
    """
    refs = schema.get("$defs") or schema.get("definitions") or {}
    cleaned = _clean_json_schema(schema, refs, 0)

    cleaned["type"] = "object"
    # 补齐形状，让输出**恒定**是 {"type", "properties", "required"} 三件套：
    # - `required` 真的会缺：pydantic 只要发现"没有任何必填参数"就整个不发这个键，
    #   而"无参数工具"和"参数全有默认值"恰恰是最常见的两种形态；
    # - `properties` 只可能缺在 `Tool.from_schema` 那条路（手写 schema）上，函数那条路 pydantic 一定发；
    # - 两行都是 `setdefault`：**键已经在了就一个字都不动**（不是赋值成空）。
    # 客户端就不必各自写 `.get("required", [])` —— 那种防御代码写漏一处，就是运行到半夜才炸的 KeyError。
    cleaned.setdefault("properties", {})
    cleaned.setdefault("required", [])
    return cleaned


def _create_model(model_name: str, fields: dict[str, tuple], tool_name: str) -> type[BaseModel]:
    """``create_model`` 的一层包装：把 pydantic 的报错换成看得懂的话。

    最常见的坑是参数名和 ``BaseModel`` 自带的属性撞名（``schema`` / ``copy`` / ``json`` /
    ``fields`` …）—— pydantic 会抛一个 NameError，看不出是谁的问题。
    """
    try:
        # ConfigDict(extra="ignore"): 忽略额外字段，不报错，也不保留
        return create_model(model_name, __config__=ConfigDict(extra="ignore"), **fields)
    except ToolBindError:
        raise
    except Exception as exc:
        raise ToolBindError(
            f"工具 {tool_name!r} 的参数模型建不出来: {exc} —— "
            "多半是某个参数名和 pydantic 的内置属性撞了（schema / json / copy / fields 等），"
            "换成别的名字（如 schema → schema_text）即可"
        ) from exc


def _build_model(func: Callable, parameters_desc: Mapping[str, str] = None) -> type[BaseModel]:
    """按签名 ＋ 类型注解造 pydantic 模型：**既是校验器，也是 schema 的来源**。

    ``parameters_desc``：``{参数名: 一句话说明}``，会挂成该属性的 ``description``。
    （也可以直接在注解里写 ``Annotated[int, Field(description="…")]``，两者等价。）
    """
    tool_name = getattr(func, "__name__", str(func))
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError) as exc:
        raise ToolBindError(f"拿不到 {tool_name!r} 的函数签名: {exc}") from exc

    # 注解可能是**字符串**：模块里一旦写了 `from __future__ import annotations`（很常见），
    # 所有注解都会变成字符串，所以必须用 get_type_hints 解析一遍。
    # include_extras=True 才保得住 Annotated[..., Field(description=…)] 里的参数说明。
    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception as exc:
        raise ToolBindError(f"{tool_name!r} 的类型注解解析不出来: {exc} —— 多半是它引用的类型在当前模块里"
                            "取不到（比如类型定义在函数内部），把它挪到模块顶层即可") from exc

    fields: dict[str, tuple] = {}
    for param_name, param in signature.parameters.items():
        if param_name in ("self", "cls"):
            continue

        # *args / **kwargs 不可能出现在 schema 里 —— 与其悄悄跳过（工具会拿到一个
        # 模型根本没被要求传的参数），不如在绑定期就报错
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            kind_label = "*args" if param.kind is inspect.Parameter.VAR_POSITIONAL else "**kwargs"
            raise ToolBindError(
                f"{tool_name!r} 的参数 {param_name!r} 是 {kind_label} —— 不能出现在 schema 里，请写成显式参数")

        annotation = hints.get(param_name, param.annotation)
        if annotation is inspect.Parameter.empty:
            raise ToolBindError(f"{tool_name!r} 的参数 {param_name!r} 没写类型注解")

        description = (parameters_desc or {}).get(param_name)
        if description:
            annotation = Annotated[annotation, Field(description=description)]

        # 没有默认值 = 必填（pydantic 用 ... 表示必填）
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param_name] = (annotation, default)

    return _create_model(f"{tool_name}_params", fields, tool_name)


def _build_parameters(validator: type[BaseModel], tool_name: str) -> dict[str, Any]:
    """校验器 → **给模型看的** JSON Schema（归一化之后）。
    """
    try:
        raw = validator.model_json_schema()
    except ToolBindError:
        raise
    except Exception as exc:
        raise ToolBindError(
            f"工具 {tool_name!r} 的 JSON Schema 生成失败: {exc} —— "
            "多半是某个参数模型（BaseModel 子类）定义在函数内部或临时作用域里，"
            "pydantic 事后重建不出来；把它挪到模块顶层即可"
        ) from exc
    return normalize_schema(raw)


_JSON_TO_PY: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _annotation_from_property(prop: Any) -> Any:
    """JSON Schema 的**一个属性** → Python 注解。认不出来就 ``Any``，绝不乱猜。"""
    if not isinstance(prop, dict):
        return Any

    # Optional[X] 在 schema 里长这样：{"anyOf": [{"type": "integer"}, {"type": "null"}]}
    for key in ("anyOf", "oneOf"):
        variants = prop.get(key)
        if isinstance(variants, list) and variants:
            annotations = [_annotation_from_property(v) for v in variants]
            if any(a is Any for a in annotations):
                return Any  # 有一个认不出，就整体放宽
            merged: Any = annotations[0]
            for extra in annotations[1:]:
                merged = merged | extra
            return merged

    json_type = prop.get("type")
    if json_type == "array":
        items = prop.get("items")
        inner = _annotation_from_property(items) if isinstance(items, dict) else Any
        return list[inner]
    if json_type in _JSON_TO_PY:
        return _JSON_TO_PY[json_type]

    if "enum" in prop and isinstance(prop["enum"], list) and prop["enum"]:
        return type(prop["enum"][0]) if all(
            isinstance(v, type(prop["enum"][0])) for v in prop["enum"]) else Any
    if "const" in prop:
        return type(prop["const"])
    return Any


def _model_from_schema(schema: Mapping[str, Any], tool_name: str) -> type[BaseModel]:
    """从 JSON Schema 反推校验器。

    认得出的类型就按类型校验；认不出的退化成 ``Any`` —— 但**仍然**会做两件最要紧的事：
    **必填键在不在**、**多余键丢掉**。
    """
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    required = set(schema.get("required") or ())

    fields: dict[str, tuple] = {}
    for param_name, prop in properties.items():
        annotation = _annotation_from_property(prop)
        default = ... if param_name in required else None
        fields[param_name] = (annotation, default)

    return _create_model(f"{tool_name}_params", fields, tool_name)


@dataclass
class Tool:
    """一个工具：**给模型看的接口**（name / description / parameters）＋ 真正干活的函数。

    工具函数就是个普通函数：参数是模型给的参数，返回字符串（或 dict / list）。
    **它不需要知道** workspace、call_id 这些东西 —— 那些由执行器和闸门在外面处理
    （老项目也是这样：内置工具用 ``make_coding_tools(workspace)`` 把工作目录闭包进去）。
    """

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable
    #: 参数校验器。**执行前一定要过它**
    validator: type[BaseModel] | None = None
    # 工具结果超过此长度会被截断——工具结果是上下文的最大污染源
    max_result_chars: int = 2000
    #: 单个工具的超时（秒）。``None`` = 用执行器的 ``default_timeout_s``；都没有 = 不设超时。
    timeout_s: float | None = None
    #: 这次调用**调的是什么**（``ToolKind.TOOL`` / ``ToolKind.SUBAGENT``）。普通工具是 tool；
    #: 子代理委派是 subagent —— 它走同一对 ``tool_start``/``tool_result`` 事件，只是 ``kind`` 不同。
    tool_type: str = ToolType.TOOL

    def __post_init__(self) -> None:
        """绑定期 fail fast：工具定义错了，别等运行到一半才发现。

        检查顺序是刻意的：**先看"这个函数根本不能当工具用"**（不可调用 / async / generator），
        再看名字与描述。否则给一个 async 函数报"缺少 description"，会把人带到错误方向。
        """
        if not callable(self.func):
            raise ToolBindError(f"工具 {self.name!r} 的 func 不可调用")
        if inspect.isgeneratorfunction(self.func) or inspect.iscoroutinefunction(self.func):
            raise ToolBindError(
                f"工具 {self.name!r} 不支持 generator / async 函数（本内核只做同步调用）"
            )
        if not NAME_PATTERN.match(self.name or ""):
            raise ToolBindError(f"工具名不合法: {self.name!r}（要求 ^[a-z_][a-z0-9_]*$）")
        if not (self.description or "").strip():
            raise ToolBindError(
                f"工具 {self.name!r} 缺少 description —— 它是写给模型的接口文档，"
                "写在 @tool(description=...) 或 docstring 里"
            )
        if not isinstance(self.parameters, dict) or self.parameters.get("type") != "object":
            raise ToolBindError(
                f"工具 {self.name!r} 的 parameters 必须是 type=object 的 JSON Schema"
            )
        if not isinstance(self.validator, type) or not issubclass(self.validator, BaseModel):
            raise ToolBindError(
                f"工具 {self.name!r} 的 validator 必须是 pydantic BaseModel 子类 —— "
                "用 @tool 装饰，或走 Tool.from_schema(...)"
            )
        if not isinstance(self.max_result_chars, int) or self.max_result_chars <= 0:
            raise ToolBindError(
                f"工具 {self.name!r} 的 max_result_chars 必须是正整数，现在是 {self.max_result_chars!r}"
            )
        if self.timeout_s is not None and (
                isinstance(self.timeout_s, bool)
                or not isinstance(self.timeout_s, (int, float))
                or self.timeout_s <= 0
        ):
            raise ToolBindError(
                f"工具 {self.name!r} 的 timeout_s 必须是正数或 None，现在是 {self.timeout_s!r}"
            )
        if self.tool_type not in (ToolType.TOOL, ToolType.SUBAGENT):
            raise ToolBindError(
                f"工具 {self.name!r} 的 kind 不合法: {self.tool_type!r}（只能是 tool / subagent）"
            )

    def validate(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """校验参数、补上默认值，返回可以直接 ``**`` 展开的字典。

        校验不过就抛 ``pydantic.ValidationError``
        """
        model = self.validator.model_validate(dict(args or {}))
        return model.model_dump()

    def invoke(self, args: Mapping[str, Any]) -> Any:
        """校验 → 调用，返回**原始**结果
        """
        return self.func(**self.validate(args))

    @classmethod
    def from_schema(cls, *, name: str, description: str, parameters: Mapping[str, Any],
                    func: Callable, max_result_chars: int = 2000,
                    timeout_s: float | None = None, tool_type: str = ToolType.TOOL) -> "Tool":

        """给"没有 Python 签名"的工具用：schema 直接给（MCP 工具、技能手册、子代理走这条）"""
        raw = dict(parameters or {})
        return cls(
            name=name,
            description=description,
            parameters=normalize_schema(raw),
            func=func,
            validator=_model_from_schema(raw, name),
            max_result_chars=max_result_chars,
            timeout_s=timeout_s,
            tool_type=tool_type,
        )


def tool(func: Callable = None, *, name: str = None, description: str = None,
         parameters_desc: Mapping[str, str] = None, max_result_chars: int = 2000, timeout_s: float | None = None):
    """装饰器：``@tool`` 或 ``@tool(name=..., max_result_chars=...)``。

    name 取函数名，description 取 docstring —— **docstring 是写给模型的接口文档**。
    ``parameters_desc`` 给单个参数补说明（``{参数名: 说明}``）。
    """

    def wrap(f: Callable) -> Tool:
        tool_name = name or getattr(f, "__name__", "")
        validator = _build_model(f, parameters_desc)
        return Tool(
            name=tool_name,
            description=description if description is not None else (f.__doc__ or "").strip(),
            parameters=_build_parameters(validator, tool_name),
            func=f,
            validator=validator,
            max_result_chars=max_result_chars,
            timeout_s=timeout_s,
        )

    return wrap(func) if func is not None else wrap


@dataclass(frozen=True)
class Deny:
    """拒绝裁决。

    ``source`` 是为了让 ``outcome`` 能区分"被权限策略拒绝"和"被钩子拦截"。
    """

    reason: str
    source: str = "policy"


class DenySource(StrEnum):
    #: 拒绝来源：被权限策略拒绝
    DENY_POLICY = "policy"
    #: 拒绝来源：被钩子拦截
    DENY_HOOK = "hook"

    ERROR = "error"


@dataclass(frozen=True)
class CallVerdict:
    """一次调用**能不能执行**的裁决（由 harness 层的判定链给出：钩子 → 权限策略）。
    """

    #: ``allow`` 直接跑；``deny`` 装配失败结果；``ask`` 走审批（见 ``approval_reason``）。
    action: str
    #: 要执行的调用（钩子改写后）。
    call: ToolCall
    #: 给模型 / 人看的一句话理由（为什么拒、为什么问）。
    reason: str = ""
    #: 谁定的：``hook`` / ``policy`` / ``unknown``。
    source: str = DenySource.DENY_POLICY
    #: 风险等级（``ask`` 时用来取超时默认值、也给人看）。
    risk: str = ""

    @property
    def is_denied(self) -> bool:
        """不许执行（``deny``）。**注意**：``ask`` 不算 —— "还没决定"和"被拒绝"不同。"""
        return self.action == "deny"

    @property
    def needs_approval(self) -> bool:
        """需要人批准（``ask``）：循环据此 yield 审批请求。"""
        return self.action == "ask"


@dataclass
class ToolResult:
    """一次工具调用的结果。

    **执行器只产出它，不碰事件**：``tool_start`` / ``tool_result`` 由 harness 层写

    ``call`` 是**真正被执行的那个调用**（判定链可能改写过参数），所以它也正是应该写进
    ``tool_start`` 的那个调用 —— "审的就是要执行的"。
    """

    call: ToolCall
    outcome: str  # 取值见 events.Outcome（闭集）
    message: ToolMessage  # 要进历史的那条；失败时 is_error=True
    duration_ms: int = 0
    truncated_from: int | None = None  # 截断前的原始长度；没截断就是 None
    spilled_to: str | None = None  # 全文落在哪（相对 workspace）
    thread_leaked: bool = False  # 超时后被"放弃等待"的线程（Python 杀不掉，只能记账）
    tool_type: str = ToolType.TOOL  # 这次调用调的是什么（tool / subagent）
    child_session: str | None = None  # tool_type=subagent 时：子会话 id（看图往下钻）
    usage: Usage | None = None  # tool_type=subagent 时：该子代理一次 loop 的总消耗


@dataclass
class SubagentOutcome:
    """子代理委派的结果（逃生口，和 ``ToolMessage`` 同级）。

    子代理工具 ``tool_type=subagent`` 的 ``func`` 返回它；执行器认出来就打包成
    ``tool_type=subagent`` 的 ``ToolResult``：

    - ``message`` → 进父历史的那条 ToolMessage（``is_error`` 表示子代理失败）；
      ``tool_call_id`` / ``name`` 由执行器补上（工具不知道自己的 call id）。
    - ``child_session`` → 指向子会话文件（看图往下钻）。
    - ``usage`` → 该子代理一次 loop 的总消耗（含它自己的压缩与嵌套子代理）。
    """

    message: ToolMessage
    child_session: str
    usage: Usage


@dataclass(frozen=True)
class ExecOptions:
    """一次批次共用的执行参数（免得每个函数都挂一长串参数）。"""

    #: 工作区：只用来决定"落盘的相对路径怎么写"以及默认落盘目录
    workspace: str | None = None
    #: 落盘目录；None = ``<workspace>/.harness/results``
    result_dir: str | None = None
    #: 工具自己没写 ``timeout_s`` 时用这个；两者都没有 = 不设超时（在当前线程直接跑）
    default_timeout_s: float | None = None


def _denied(call: ToolCall, deny: Deny, duration_ms: int) -> ToolResult:
    """把裁决变成结果：来源决定 ``outcome``（策略拒绝 ≠ 工具本身出错）。"""
    outcome = Outcome.BLOCKED if deny.source == DenySource.DENY_HOOK else Outcome.DENIED
    if deny.source == DenySource.ERROR:
        outcome = Outcome.FAILED
    return _failure(call, outcome, f"[已拒绝] {deny.reason}", duration_ms)


def _failure(call: ToolCall, outcome: str, content: str, duration_ms: int = 0,
             tool_type: str = ToolType.TOOL) -> ToolResult:
    """装配校验失败的工具结果"""

    message = ToolMessage(content=content, tool_call_id=call.id, name=call.name, is_error=True)
    return ToolResult(call=call, outcome=outcome, message=message, duration_ms=duration_ms, tool_type=tool_type)


def _ms(started: float) -> int:
    """给定开始事件，返回直到当前为止的时间差，单位：毫秒"""

    return int((time.monotonic() - started) * 1000)


#: pydantic 的常见错误 → 给模型看的一句话。认不出的就照原样用它的英文 msg（不猜）。
_VALIDATION_REASONS = {
    "missing": "缺少这个参数",
    "int_parsing": "不是整数",
    "int_type": "不是整数",
    "float_parsing": "不是数字",
    "float_type": "不是数字",
    "bool_parsing": "不是布尔值",
    "bool_type": "不是布尔值",
    "string_type": "不是字符串",
    "list_type": "不是数组",
    "dict_type": "不是对象",
    "model_type": "不是对象",
    "json_invalid": "不是合法 JSON",
}


def _format_validation_error(exc: ValidationError) -> str:
    """把 pydantic 的报错压成"一行一条"，不带它的噪音尾巴。

    形如：``limit: 不是整数（收到 'abc'）``。
    pydantic 原始的 ``[type=int_parsing, input_value='abc', input_type=str]`` 又长又难读，
    而这条消息是**直接喂给模型**的 —— 它越干净，模型越容易一次改对。
    """
    lines = ["[错误] 参数不对："]
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ())) or "(参数)"
        error_type = error.get("type", "")
        reason = _VALIDATION_REASONS.get(error_type, error.get("msg", "不合法"))
        if "input" in error:
            reason = f"{reason}（收到 {error['input']!r}）"
        lines.append(f"  {location}: {reason}")
    return "\n".join(lines)


def _call_tool(call: ToolCall, tool_obj: Tool, options: ExecOptions, started: float) -> ToolResult:
    """校验参数 → 调用工具函数 → 结果处理。"""

    token = set_call(CallContext(call_id=call.id, name=call.name))
    try:
        raw = tool_obj.invoke(call.args)  # 内部先过 validator（脏参数一次都不进函数）
    except ValidationError as exc:
        return _failure(call, Outcome.FAILED, _format_validation_error(exc), _ms(started), tool_obj.tool_type)
    except ToolBindError as exc:
        return _failure(call, Outcome.FAILED, f"[错误] {exc}", _ms(started), tool_obj.tool_type)
    except Exception as exc:
        # 获取当前未处理异常的完整调用栈文本，但只保留最后 2 层。
        detail = traceback.format_exc(limit=2)
        return _failure(call, Outcome.FAILED, f"[错误] 工具执行失败: {type(exc).__name__}: {exc}\n{detail}",
                        _ms(started), tool_obj.tool_type)
    finally:
        reset_call(token)
    return _build_result(call, tool_obj, raw, options, started)


def _to_text(result: Any) -> str:
    """工具返回值 → 给模型看的文本

    dict / list 用 ``json.dumps`` 而不是 ``str(...)``：后者是 Python repr（单引号、
    True/False 大写）。
    """
    if isinstance(result, str):
        return result
    if isinstance(result, (dict, list)):
        return json.dumps(result, ensure_ascii=False, default=str)
    if result is None:
        return "(无输出)"
    return str(result)


def _truncate(text: str, limit: int, call: ToolCall,
              options: ExecOptions) -> tuple[str, int | None, str | None]:
    """超长则：全文落盘 ＋ 只给模型前 N 个字符 ＋ 一句"完整内容在哪"。
    """
    if limit <= 0 or len(text) <= limit:
        return text, None, None

    total_lines = text.count("\n") + 1
    hint = f"完整内容 {len(text)} 字符 / {total_lines} 行"
    relative_path = _spill(text, call, options)

    if relative_path is None:
        tail = f"\n…[结果已截断：{hint}，写盘失败，只能给出前面这些]"
    else:
        tail = f"\n…[结果已截断：{hint}，已存到 {relative_path}，可用 read_file 读取]"

    return text[:limit] + tail, len(text), relative_path


def _spill(text: str, call: ToolCall, options: ExecOptions) -> str | None:
    """把全文写到结果目录，返回**给模型看的相对路径**；失败返回 ``None``（不抛）。

    降级行为是刻意的：写盘失败只影响"模型能不能取回全文"，不该把一次成功的工具调用
    变成失败。
    """
    try:
        if options.result_dir:
            directory = Path(options.result_dir)
        elif options.workspace:
            directory = Path(options.workspace) / DEFAULT_RESULT_DIR
        else:
            return None  # 没给落盘位置：只能截断了

        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{call.id}.txt"
        path.write_bytes(text.encode("utf-8")[:MAX_RESULT_FILE_BYTES])

        if options.workspace:
            try:
                return os.path.relpath(path, options.workspace)
            except ValueError:  # 跨盘（Windows 上真会发生）
                return str(path)
        return str(path)
    except (OSError, ValueError):  # 无权限 / 磁盘满 / 路径含 NUL ……
        return None


def _build_result(call: ToolCall, tool_obj: Tool, raw: Any, options: ExecOptions,
                  started: float) -> ToolResult:
    """序列化 → 截断落盘 → 打包成 ToolResult。"""
    # 逃生口 1：子代理结果（tool_type=subagent）—— 携带 child_session 与总消耗
    if isinstance(raw, SubagentOutcome):
        message = replace(raw.message, tool_call_id=call.id, name=call.name)
        outcome = Outcome.FAILED if message.is_error else Outcome.OK
        return ToolResult(call=call, outcome=outcome, message=message, duration_ms=_ms(started),
                          tool_type=tool_obj.tool_type, child_session=raw.child_session, usage=raw.usage)

    # 逃生口 2：工具直接返回一个 ToolMessage（自设 is_error / 自定义内容）。
    # ``tool_call_id`` / ``name`` 由执行器补上 —— 工具不知道自己的 call id，不该要求它填对。
    if isinstance(raw, ToolMessage):
        message = replace(raw, tool_call_id=call.id, name=call.name)
        outcome = Outcome.FAILED if message.is_error else Outcome.OK
        return ToolResult(call=call, outcome=outcome, message=message, duration_ms=_ms(started),
                          tool_type=tool_obj.tool_type)

    text = _to_text(raw)
    text, truncated_from, spilled_to = _truncate(text, tool_obj.max_result_chars, call, options)
    message = ToolMessage(content=text, tool_call_id=call.id, name=call.name)
    return ToolResult(call=call, outcome=Outcome.OK, message=message, duration_ms=_ms(started),
                      truncated_from=truncated_from, spilled_to=spilled_to, tool_type=tool_obj.tool_type)


def execute_tool_call(call: ToolCall, tools_by_name: dict[str, Tool],
                      options: ExecOptions) -> ToolResult:
    """执行**一条**调用 —— 执行器的最小入口（也是唯一的入口）。

    **任何失败都降级成 is_error 的 ToolResult，绝不往外抛** —— 模型看到错误之后可以自己
    纠正（换参数 / 换工具 / 求助），循环不会因为一个工具炸掉。
    """
    started = time.monotonic()

    # 查工具表（判定已经由调用方做过了，这里只管"名字存不存在"）
    tool_obj = tools_by_name.get(call.name)
    if tool_obj is None:
        return _failure(call, Outcome.FAILED, f"[错误] 不存在名为 {call.name!r} 的工具", _ms(started))

    timeout = tool_obj.timeout_s if tool_obj.timeout_s is not None else options.default_timeout_s
    if timeout is None:
        # 没设超时：直接在**当前线程**跑（不起线程，最省）
        return _call_tool(call, tool_obj, options, started)

    # 设了超时：丢进一个 **daemon 线程**，到点放弃等待。
    # - Python 杀不掉线程 → 只能"放弃等待"，并把这次记账（thread_leaked）
    # - daemon=True 很关键：卡死的工具**不会拖住解释器退出**
    #   （线程池做不到 —— 它的线程在解释器退出时会被 join）
    box: list[ToolResult] = []

    def target() -> None:
        try:
            box.append(_call_tool(call, tool_obj, options, started))
        except BaseException as exc:  # 兜底：执行器自己不能再炸
            box.append(
                _failure(call, Outcome.FAILED, f"[错误] 执行器内部错误: {exc!r}", _ms(started), tool_obj.tool_type))

    worker = threading.Thread(target=target, daemon=True, name=f"tool-{call.name}-{call.id}")
    worker.start()
    worker.join(timeout)

    if worker.is_alive():
        result = _failure(call, Outcome.TIMEOUT, f"[错误] 工具执行超时（{timeout} 秒），已放弃等待",
                          _ms(started), tool_obj.tool_type)
        result.thread_leaked = True
        return result
    return box[0]


# ---------------------------------------------------------------------------
# 拒绝结果的装配（判定链的产物 → 进历史的那条 ToolMessage）
# ---------------------------------------------------------------------------
def denied_result(call: ToolCall, reason: str, source: str = DenySource.DENY_POLICY,
                  risk: str = "") -> ToolResult:
    """把"不许执行"装配成一条结果（**工具函数一次都不会进**）。

    ``source`` 决定 ``outcome``：

    - ``hook`` → ``blocked_by_hook``（被钩子拦）；
    - 其余（``policy`` / ``unknown`` / 审批被拒 / 没通道 / 超时兜底）→ ``denied_by_policy``。
    """
    outcome = Outcome.BLOCKED if source == DenySource.DENY_HOOK else Outcome.DENIED
    return _failure(call, outcome, f"[已拒绝] {reason}", 0,
                    ToolType.SUBAGENT if risk == ToolType.SUBAGENT else ToolType.TOOL)
