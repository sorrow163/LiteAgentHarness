import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from src.core.context import ContextManager
from src.core.events import EventType, DataKey, StopReason, Outcome, Event, ToolType, get_new_id
from src.core.loop import Agent
from src.core.message import Usage, ToolMessage
from src.core.models import BaseChatModel
from src.core.runtime import get_run, get_call
from src.core.tool import Tool, SubagentOutcome

#: 子代理 system prompt 追加段：约束它别把问题抛回给用户。
_NO_ASK_BACK = "\n\n除非父代理明确要求，否则不要把问题抛回给用户；有疑问就把结论与疑问一起返回给父代理。"


def _next_child_id(store: Any, parent_session_id: str) -> str:
    """这个父会话下**第一个没被占用的** ``.sub-N``。

    ★ 为什么不能用一个"从 1 数上去"的内存计数器：``task`` 工具**每次装配 agent 都会重建**
    （`make_task_tool` 里的状态活不过一次 `_build_agent`），计数器于是跟着从 1 重来 ——
    而这个会话**上一次已经派过**的 ``.sub-1`` 还在磁盘上，`store.create` 对已存在的 id 是
    **明确拒绝**（拒绝覆盖）。后果就是"重新打开会话之后再委派"直接失败：
    在一个会话里派过一次、退出、`--session` 接回来、再派一次 → 撞名。

    所以编号从**文件**里推：已有 ``.sub-1`` / ``.sub-3`` 就取 ``.sub-2``。
    id 的形状（``<parent>.sub-N``）保持不变 —— 它是 `child_session`、文件头的 `parent`
    与文档里都在用的约定，改成时间戳会让那串 id 到处都难读。
    """
    taken = set(store.child_ids(parent_session_id))
    n = 1
    while f"{parent_session_id}.sub-{n}" in taken:
        n += 1
    return f"{parent_session_id}.sub-{n}"

@dataclass
class SubagentDef:
    """一个可被委派的子代理档案。``description`` 是写给**主模型**看的：
    它靠这段话决定"什么任务派给谁"—— 和工具的 ``description`` 同一个道理。"""

    name: str
    description: str
    system_prompt: str
    tools: Sequence[Tool] = ()
    model: BaseChatModel | None = None      # None = 沿用主模型的模型
    max_turns: int = 20
    max_context_tokens: int = 40_000

def make_task_tool(subagents: Sequence[SubagentDef], default_model: BaseChatModel, *,
                   store, parent_session_id: str, meta: dict[str, Any] | None = None,
                   judge=None, summarizer: BaseChatModel | None = None,
                   max_depth: int = 2, _depth: int = 0) -> Tool:
    """构造 ``task`` 工具：把子任务委派给专职子代理，返回其最终结论。

    ``store`` / ``parent_session_id`` / ``meta`` 是子会话隔离的原材料：每次委派都建一个
    ``<parent>.sub-N.jsonl`` 文件（``head.parent = parent_session_id``）。深度控制是**静态**的
    —— 只有 ``depth+1 < max_depth`` 时，子代理的工具箱里才会再放一把 ``task``。

    **子代理内部不审批**：给它 ``judge``（判定链仍然生效 —— 模式／风险照样管着它），
    但**不给审批入口**。于是子代理遇到 ``ask`` 时按"没有人可问"处理（``by="none"`` →
    按风险等级默认值：写 / 执行拒绝）。理由：审批请求要从子会话冒到**父**消费者那里，
    那得把整条执行链变成生成器；而"自动化委派中途弹个窗"本身也不是想要的交互。
    """
    registry: dict[str, SubagentDef] = {s.name: s for s in subagents}
    roster = "\n".join(f"- {s.name}: {s.description}" for s in subagents)

    def task(agent: str, prompt: str):
        spec = registry.get(agent)
        if spec is None:
            return f"[错误] 没有名为 {agent!r} 的子代理，可用: {', '.join(registry) or '(无)'}"

        run_ctx = get_run()                  # 父 run 的信息（run_id / workspace / memory …）
        call_ctx = get_call()                # 本次调用的 call_id / name
        started = time.monotonic()

        # 子会话：独立文件，head.parent = 父会话 id
        child_id = _next_child_id(store, parent_session_id)

        # 深度没到底 → 子代理的工具箱里再放一把 task（指向同一批子代理，深度 +1）
        tools = list(spec.tools)
        if _depth + 1 < max_depth and len(registry) > 0:
            tools.append(make_task_tool(subagents, default_model, store=store,
                                        parent_session_id=child_id, meta=meta,
                                        judge=judge, summarizer=summarizer,
                                        max_depth=max_depth, _depth=_depth + 1))


        child_meta = dict(meta or {})
        child_meta["agent"] = spec.name
        child = store.create(child_id, parent=parent_session_id, meta=child_meta)

        sub = Agent(
            model=spec.model or default_model,
            tools=tools,
            system_prompt=spec.system_prompt + _NO_ASK_BACK,
            max_turns=spec.max_turns,
            context=ContextManager(max_context_tokens=spec.max_context_tokens),
            judge=judge,
            # ★ 无人可问：审批请求就地按默认值裁决，不 yield（见上面的说明）
            approval_resolver=lambda request: None,
            summarizer=summarizer,
            name=f"{spec.name}@{_depth + 1}",
            stream_text=False,               # 子代理不推 text_delta 给父（⚠X 同款理由）
        )

        # 子文件结构：session_start → 根 tool_start(kind=subagent) → run → 根 tool_result → session_end
        session_span = get_new_id()
        root_span = get_new_id()
        child.write_event(Event(type=EventType.SESSION_START, span=session_span))
        child.write_event(Event(
            type=EventType.TOOL_START, span=root_span, parent_span=session_span,
            run_id=run_ctx.run_id,           # ⚠K2：子事件带**父 run 的 run_id**
            data={DataKey.CALL_ID: call_ctx.call_id, "name": call_ctx.name,
                  "args": {"agent": agent, "prompt": prompt},
                  DataKey.TOOL_TYPE: ToolType.SUBAGENT,
                  "system_prompt": sub.system_prompt},
        ))

        # 跑子代理，事件全落子文件；顺带抓"最终回答 / 结束原因 / 用量"
        final = "(子代理没有产出文本)"
        stop_reason = StopReason.ERROR
        total = Usage()
        for event in sub.run(prompt, run_id=run_ctx.run_id, session_span=root_span,
                             workspace=run_ctx.workspace, result_dir=run_ctx.result_dir,
                             memory=run_ctx.memory):
            if event.type == EventType.ASSISTANT_MESSAGE:
                message = event.data[DataKey.MESSAGE]
                if not message.tool_calls:    # 无工具调用 = 最终回答
                    final = message.content or ""
            if event.type == EventType.RUN_END:
                total = event.data.get(DataKey.USAGE, Usage())
                stop_reason = event.data.get(DataKey.STOP_REASON, StopReason.ERROR)
            child.write_event(event)

        # 子代理的总消耗 = 子 run 的 ``RUN_END.usage``。**它已经是那棵子树的全量**
        # （子代理自己的调用 ＋ 压缩 ＋ 它再委派的下级），所以下级那一块不能再加一遍 ——
        # 加了就是同一个孙代理被计两次。
        duration_ms = int((time.monotonic() - started) * 1000)

        # 只有正常收尾（end_turn）才算成功；max_turns / stalled / error 都要让父模型知道
        ended_ok = stop_reason == StopReason.END_TURN
        outcome = Outcome.OK if ended_ok else Outcome.FAILED
        conclusion = ToolMessage(
            content=final if ended_ok else f"[子代理以 {stop_reason} 结束] {final}",
            tool_call_id=call_ctx.call_id, name=call_ctx.name, is_error=not ended_ok,
        )

        child.write_event(Event(
            type=EventType.TOOL_RESULT, span=root_span, run_id=run_ctx.run_id,
            data={DataKey.DURATION_MS: duration_ms, DataKey.OUTCOME: outcome,
                  DataKey.TOOL_TYPE: ToolType.SUBAGENT, DataKey.USAGE: total}))
        child.write_event(Event(type=EventType.SESSION_END, span=session_span, data={"reason": "done"}))

        return SubagentOutcome(message=conclusion, child_session=child.id, usage=total)

    return Tool.from_schema(
        name="task",
        description=(
            "把一个独立子任务委派给专职子代理执行，返回其最终结论。"
            "适用于：需要大量探索/阅读但主对话只需要结论的任务；"
            "可以在一轮里发多个 task 调用并行推进互不依赖的子任务。\n"
            "agent: 子代理名字，必须是下列之一\n" + roster + "\n"
            "prompt: 交给子代理的完整任务描述（它看不到当前对话，必须自包含）"
        ),
        parameters={"type": "object",
                    "properties": {"agent": {"type": "string"}, "prompt": {"type": "string"}},
                    "required": ["agent", "prompt"]},
        func=task,
        max_result_chars=32_000,             # 结论是子代理的产出，别按 2000 拦腰截断
        tool_type=ToolType.SUBAGENT,
    )