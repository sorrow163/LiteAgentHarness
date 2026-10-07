import hashlib
import json
import queue
import time
import copy
from collections.abc import  Callable, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from dataclasses import dataclass
from typing import Sequence, Any


from src.core.context import ContextManager
from src.core.errors import ToolBindError
from src.core.events import Event, get_new_id, EventType, DataKey, StopReason, Outcome, CompactionStrategy, \
    ApprovalRequest, now_ms, ApprovalSource, ApprovalDecision, ErrorReason, ToolType
from src.core.message import Messages, AIMessage, to_jsonable, HumanMessage, Usage, SystemMessage, ToolCall
from src.core.models import BaseChatModel
from src.core.runtime import RunContext, set_run, reset_run
from src.core.tool import Tool, ToolResult, ExecOptions, execute_tool_call, CallVerdict, DenySource, denied_result


def _digest(messages: Messages) -> str:
    """校验用指纹：证明"重放出的历史 == 实际发出去的历史"。"""
    payload = json.dumps([to_jsonable(msg) for msg in messages], ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _turn_signature(message: AIMessage) -> tuple:
    """这一轮"干了什么"的指纹，给无进展检测用（连续两轮一样 = 卡死）。"""
    calls = tuple((call.name, json.dumps(call.args, ensure_ascii=False, sort_keys=True, default=str))
        for call in (message.tool_calls or []))
    return message.content or "", calls


def _stall_event(run_id: str, run_span: str, turn: int, reason: str) -> Event:
    return Event(type=EventType.ERROR, run_id=run_id, turn=turn, span=run_span,
                 data={DataKey.ERROR_SOURCE: ErrorReason.INTERNAL,
                       "message": f"[卡死判定] {reason}",
                       "recoverable": False})


def _compaction_event(result, run_id: str) -> Event:
    data = {"strategy": result.strategy,
            "before_tokens": result.before_tokens,
            "after_tokens": result.after_tokens,
            DataKey.DURATION_MS: result.duration_ms}
    if result.strategy == CompactionStrategy.CLEAR_TOOL_RESULTS:
        data.update(cleared_call_ids=list(result.cleared_call_ids),
                    cleared_chars=result.cleared_chars,
                    saved_tokens_estimate=result.saved_tokens_estimate)
    else:   # summarize
        data.update(kept_count=result.kept_count,
                    dropped_messages=result.dropped_messages)
        if result.recap is not None:
            data["recap"] = result.recap
        if result.summarizer_model is not None:
            data["summarizer_model"] = result.summarizer_model
        if result.summarizer_usage is not None:
            data["summarizer_usage"] = result.summarizer_usage
    return Event(type=EventType.COMPACTION, run_id=run_id, span=get_new_id(), data=data)


#: 一轮里最多**同时**跑几个委派。多出来的在池子里排队。
#:
#: 它不是配置项，因为"并发"是**委派这种调用的语义**（一轮里发多个委派，就是模型在说这几件事
#: 互不依赖），不需要谁去开关它。但池子必须有个上限：一轮里最多能发 `max_tool_calls_per_turn`
#: （默认 32）个调用，全是委派的话就是 32 条线程同时各跑一个完整 loop、各自再调模型。
MAX_CONCURRENT_DELEGATIONS = 8


@dataclass
class Agent:
    """一个可运行的 agent = 模型 + 工具 + 指令 +（可选的）上下文策略与闸门。"""

    model: BaseChatModel
    summarizer: BaseChatModel   #: AI 总结压缩用的模型。**必须是"不绑工具"的实例**
    tools: Sequence[Tool] | dict[str, Tool] | None = None
    system_prompt: str = ""
    max_turns: int = 40  # 安全阀：防失控循环（对应 recursion_limit）
    context: ContextManager = None

    #: **判定链**：`judge(call) -> CallVerdict`（钩子 → 权限策略，由 harness 层组装）。
    #: 它**不阻塞、不问人** —— 只说"能不能跑 / 要不要问"；问人由本循环负责。
    judge: Callable[[ToolCall], CallVerdict] = None

    #: **超时/无人回答时该给什么默认决定**（由权限策略提供：只读放行，其余拒绝）。
    #: 返回值给审批请求的 ``proposed`` 字段（"倒计时结束后会怎样"），也是驱动方 ``send(None)`` 时的默认答案。
    propose: Callable[[ToolCall], bool] = None

    #: **无人可问时的替身**：`resolve(request) -> ApprovalDecision`。
    #: 给了它就**不 yield 审批请求**，就地取答案 —— 子代理用它（子代理内部不审批）。
    approval_resolver: Callable[[ApprovalRequest], Any] = None

    #: 判定/审批的记账出口：`record(tool_type, invocation, action, by, reason)`。
    #: 内核只负责**调用**它（内核不认识"审计"），写不写、写哪去由 harness 决定。
    audit: Callable[..., None] = None

    stream_text: bool = True  # False 则整段返回（省事件量）
    name: str = "agent"
    #: 无进展检测（L1）的两条判据
    max_identical_turns: int = 3  # 连续多少轮"完全相同的回复"就判卡死
    max_tool_calls_per_turn: int = 32  # 单轮工具调用超过这个数 = 工具风暴
    _interrupted = False

    def __post_init__(self) -> None:
        # 绑定期 fail fast：重名工具在注册时就报错，不静默跳过
        seen: set[str] = set()
        for tool in self.tools:
            if tool.name in seen:
                raise ToolBindError(f"工具名重复: {tool.name!r}（注册表里已经有了）")
            seen.add(tool.name)
        self._interrupted = False
        self._inbox: queue.Queue[str] = queue.Queue()  # 中途插话（steering）用的队列

        self.model = copy.copy(self.model).bind_tools(list(self.tools))
        self.tools = {tool.name: tool for tool in self.tools}

        #: 上一次发出去的 ``sys_prompt`` 指纹（``None`` = 这条会话还没发过）。
        #: 见 :meth:`_sys_prompt_event`变了才发。
        self._sent_sys_prompt: str | None = None

    def interrupt(self) -> None:
        """请求停止。在下一个消息边界生效（协作式，不会撕裂历史）。"""
        self._interrupted = True

    def inject(self, text: str) -> None:
        """中途插入一条用户消息（steering）。在下一轮模型调用前生效，同样会
        落成一条 ``user_message`` 事件、进入历史。队列是线程安全的，TUI 线程可调。"""
        self._inbox.put(text)

    # ---- 装配与 sys_prompt ----
    def _working_messages(self, messages: Messages, memory: str) -> Messages:
        """本轮真正发给模型的列表。

        抽成一个方法是因为**两处要用同一份规则**：主循环每轮装配一次，而 ``sys_prompt``
        事件要记下"这次 run 发给模型的 prompt 长什么样"。两处各写一份的话，记下来的
        和真发出去的可能慢慢就不一样了 —— 那等于 trace 在骗人。
        """
        if self.context is not None:
            return self.context.assemble(self.system_prompt, messages, memory=memory)
        if self.system_prompt:
            return [SystemMessage(content=self.system_prompt)] + list(messages)
        return list(messages)

    def _assembled_system_prompt(self, memory: str) -> str:
        """拼出来要发给模型的**那整段 system prompt**（含记忆与 extra_context）。

        契约要求 ``sys_prompt`` 记的是这个，而不是 ``self.system_prompt`` 那个原始字段 ——
        少了记忆和 extra_context，事后按 trace 重放会得到另一份 prompt。
        """
        assembled = self._working_messages([], memory)
        return assembled[0].content if assembled else ""

    def _sys_prompt_event(self, *, run_id: str, session_span: str | None, memory: str,
                          memory_sources: Sequence[str]) -> Event | None:
        """要不要发一条 ``sys_prompt``（契约 ⚠A：**位置语义、变了才发**）。

        规则：某次 run 的 prompt ＝ 它**之前最后一条** ``sys_prompt`` 的值。所以
        "会话第一条 run 之前"发一次，之后每次 run 前比对、**变了**才追加 —— 每次都发的话，
        20KB 的 AGENTS.md × 50 次 run ＝ 1MB 重复。

        ``span`` 用会话 span：它不是某次操作（不在 ``OPENING`` / ``CLOSING`` 里），
        而是"这条会话当时看到的静态上下文"；``parent_span`` 必须留空（⚠SP 护栏会拦）。
        """
        text = self._assembled_system_prompt(memory)
        fingerprint = repr((text, tuple(self.tools)))
        if fingerprint == self._sent_sys_prompt:
            return None
        self._sent_sys_prompt = fingerprint
        return Event(type=EventType.SYS_PROMPT, run_id=run_id, span=session_span,
                     data={"system_prompt": text,
                           "tools": list(self.tools),
                           "memory_sources": list(memory_sources)})

    # ---- 主循环 ----
    def run(self, user_input: str, *, prior_messages: Messages = None,
            run_id: str = None, session_span: str = None, memory: str = "",
            memory_sources: Sequence[str] = (),  workspace: str = None, result_dir: str = None,
            default_timeout_s: float = None) -> Generator[Event]:
        """执行一次 run，逐条 ``yield`` 事件。

        一次 run = "用户说一句话 → agent 给出最终回答"。**结论不从这里返回**：
        取最后一条 ``assistant_message`` 事件即可。``run()`` 是生成器。

        ``session_span`` 是**会话 span** 的 id（``run_start`` 的 ``parent_span`` 指向它）；
        由会话层传入。``memory`` / ``memory_sources`` / ``workspace`` / ``result_dir`` 也都由
        外层给，这一层只转发（``memory_sources`` 只用于 ``sys_prompt`` 事件 —— 内容在
        ``memory`` 里，路径得单独给，否则事后从文本里认不出出处）。

        这里先设置**运行期上下文**（子代理的 ``task`` 工具靠 ``runtime.get_run()`` 读到
        ``run_id`` / ``workspace`` 等，见 core/runtime.py），再交给 ``_run`` 干正事。
        """
        effective_run_id = run_id or get_new_id()
        token = set_run(RunContext(run_id=effective_run_id, session_span=session_span,
                                   workspace=workspace, result_dir=result_dir, memory=memory))

        try:
            yield from self._run(user_input, prior_messages=prior_messages,
                                   run_id=effective_run_id, session_span=session_span,
                                   memory=memory, memory_sources=memory_sources,
                                   workspace=workspace, result_dir=result_dir,
                                   default_timeout_s=default_timeout_s)
        finally:
            reset_run(token)

    def _run(self, user_input: str, *, prior_messages: Messages = None,
             run_id: str = None, session_span: str = None, memory: str = "",
             memory_sources: Sequence[str] = (),
             workspace: str = None, result_dir: str = None,
             default_timeout_s: float = None) -> Generator[Event]:

        started = time.monotonic()
        self._interrupted = False
        run_id = run_id or get_new_id()  # 外层 run() 已默认过；这里再兜一次直接调 _run 的情况
        run_span = get_new_id()

        messages: Messages = list(prior_messages or [])
        new_input = HumanMessage(content=user_input)
        messages.append(new_input)

        # ``sys_prompt`` **在 run_start 之前**。变了才发，所以多数
        # run 这里什么都不发 —— 见 `_sys_prompt_event`。
        sys_prompt_event = self._sys_prompt_event(run_id=run_id, session_span=session_span,
                                                  memory=memory, memory_sources=memory_sources)
        if sys_prompt_event is not None:
            yield sys_prompt_event

        yield Event(type=EventType.RUN_START, run_id=run_id, span=run_span, parent_span=session_span,
                    data={"model": getattr(self.model, "model", type(self.model).__name__)})

        yield Event(type=EventType.USER_MESSAGE, run_id=run_id, span=get_new_id(), data={DataKey.MESSAGE: new_input})

        total_usage = Usage()
        stop_reason = StopReason.MAX_TURNS  # 兜底：跑到 max_turns 都没停，就是这个
        identical_streak = 0
        last_signature = None
        turns_used = 0        # 这次 loop 实际进了几轮（run_end 上要记它，见下面）

        for turn in range(1, self.max_turns + 1):
            turns_used = turn
            # ① 消息边界：先处理外部控制（中途插话 / 打断）
            while not self._inbox.empty():
                injected = HumanMessage(content=self._inbox.get())
                messages.append(injected)
                yield Event(type=EventType.USER_MESSAGE, run_id=run_id, span=get_new_id(), data={DataKey.MESSAGE: injected})
            if self._interrupted:
                stop_reason = StopReason.INTERRUPTED
                break

            # ② 装配本轮真正发给模型的列表（记忆由外层每次 run 传一次，assemble 是纯函数）
            working = self._working_messages(messages, memory)

            if self.context is not None:
                if self.context.should_compact(working, self.model.tools_schema):
                    compacted = self._compact(working)
                    if compacted is not None:
                        messages = compacted.messages
                        yield _compaction_event(compacted, run_id)
                        if getattr(compacted, "summarizer_usage", None) is not None:
                            total_usage = total_usage + compacted.summarizer_usage

            # ③ 模型决策（流式：先增量后全量）
            model_span = get_new_id()
            yield Event(type=EventType.MODEL_START, run_id=run_id, turn=turn,
                span=model_span, parent_span=run_span,
                data={"model": getattr(self.model, "model", type(self.model).__name__),
                      "message_count": len(working),
                      "digest": _digest(working),
                      "input_chars": sum(len(m.content or "") for m in working)},
            )
            call_started = time.monotonic()
            ttft_ms = None
            chunk_count = 0
            ai: AIMessage | None = None
            try:
                if self.stream_text:
                    for item in self.model.stream(working):
                        if isinstance(item, AIMessage):
                            ai = item
                        else:
                            if ttft_ms is None:
                                ttft_ms = int((time.monotonic() - call_started) * 1000)
                            chunk_count += 1
                            if isinstance(item, dict) and EventType.TEXT_DELTA in item:
                                yield Event(type=EventType.TEXT_DELTA, run_id=run_id, turn=turn,
                                            span=model_span, data={"text": item[EventType.TEXT_DELTA]})

                            if isinstance(item, dict) and EventType.REASONING in item:
                                yield Event(type=EventType.REASONING, run_id=run_id, turn=turn,
                                            span=model_span, data={"text": item[EventType.REASONING]})

                    assert ai is not None, "stream() 必须以完整 AIMessage 收尾"
                else:
                    ai = self.model.invoke(working)
            except Exception as exc:
                yield Event(type=EventType.ERROR, run_id=run_id, turn=turn, span=model_span,
                            data={DataKey.ERROR_SOURCE: ErrorReason.PROVIDER,
                                  "message": f"{type(exc).__name__}: {exc}", "recoverable": False})
                stop_reason = StopReason.ERROR
                break

            messages.append(ai)
            total_usage = total_usage + ai.usage
            yield Event(
                type=EventType.ASSISTANT_MESSAGE, run_id=run_id, turn=turn, span=model_span,
                data={DataKey.MESSAGE: ai,
                      DataKey.DURATION_MS: int((time.monotonic() - call_started) * 1000),
                      DataKey.TTFT_MS: ttft_ms or 0,
                      DataKey.CHUNK_COUNT: chunk_count,
                      DataKey.OUTCOME: Outcome.OK},
            )

            # ④ 终止判据：没有工具调用 = 模型给出最终回答
            if not ai.tool_calls:
                stop_reason = StopReason.END_TURN
                break

            # ⑤ 无进展检测（L1）：工具风暴 / 连续相同回复
            if len(ai.tool_calls) > self.max_tool_calls_per_turn:
                yield _stall_event(run_id, run_span, turn,
                                        f"单轮工具调用 {len(ai.tool_calls)} 次，超过上限: {self.max_tool_calls_per_turn}")
                stop_reason = StopReason.STALLED
                break

            signature = _turn_signature(ai)
            identical_streak = identical_streak + 1 if signature == last_signature else 1
            last_signature = signature
            if identical_streak >= self.max_identical_turns:
                yield _stall_event(run_id, run_span, turn, f"连续 {identical_streak} 轮做出完全相同的回复，疑似卡死")
                stop_reason = StopReason.STALLED
                break

            # ⑥ 逐条：**判定 → （必要时）问人 → 执行**，边跑边发事件
            #
            #    这一段是"审批与执行分开"的落点：
            #
            #    ① `tool_start` 在**派发之前**发出去。它开一个 span，而 span 的"开始时刻"
            #       就是事件的 `ts` —— 若等整批跑完再补发，轨迹里的开始时刻就成了"结束时刻"
            #       （算不出延迟），TUI 也看不见"执行中"（60 秒的命令全程无事件）。
            #    ② `judge()`（钩子 → 权限策略，纯函数、不阻塞）给出裁决：放行 / 拒绝 / **要问人**。
            #    ③ 要问人时 **yield 一个 `approval_request` 事件**，消费者的 `send(决定)`
            #       回到这里 —— 这就是"人机交互"，它发生在**本生成器自己的帧**里，
            #       所以决定能直达（不需要线程、不需要信箱）。
            #       驱动方必须用 `while + gen.send(...)`（`for` 等价于 `send(None)`，
            #       会永远拿到"没有决定"）。
            #    ④ 拒绝 → 装配 `denied_by_policy` / `blocked_by_hook` 的结果，**工具一次都不跑**；
            #       放行 → `execute_tool_call` 只认"跑哪个工具、什么参数"。
            #
            #    `approval_resolver`：给了它就**不 yield**、就地取答案（子代理用它 —— 子代理内部不审批，一律按"没有人可问"处理）。
            exec_options = ExecOptions(workspace=workspace, result_dir=result_dir, default_timeout_s=default_timeout_s)
            # 这一批调用分两路：**委派**（`task`）并发跑，其余一律串行。
            #
            # 为什么委派要并发：一轮里发多个委派，本来就是模型在说"这几件事互不依赖"，串行跑
            # 等于把它们等网络的时间叠起来。
            # 为什么其余的仍然串行：普通工具大多在改工作区（写文件、跑命令），并发会让两个工具
            # 同时动同一份东西，或者让一个工具读到别人刚写了一半的文件。那不是线程安全问题，
            # 是语义顺序问题，加锁也解决不了。
            #
            # 事件顺序有一条硬约束：**写进文件的 `tool_result` 顺序要与 `ai.tool_calls` 的顺序
            # 一致**，因为 `messages` 的顺序由它决定，而重放与下一轮真正发出去的历史都靠它。
            # 所以从**第一个真的被派出去的委派**往后，结果先攒着，等委派跑完再按调用顺序统一发；
            # 委派之前那些照旧发完就发。没有委派的批次（绝大多数）`hold_from` 一直是 `None`，
            # 走的就是老路：跑一条发一条，界面逐条完成。
            held: list[tuple[int, ToolResult, str]] = []
            pending: list[tuple[int, ToolCall, str]] = []
            hold_from: int | None = None

            for index, tool_call in enumerate(ai.tool_calls):
                # ① 开 span：**派发之前**
                tool_span = get_new_id()
                yield Event(type=EventType.TOOL_START, run_id=run_id, turn=turn, span=tool_span, parent_span=run_span,
                    data={DataKey.CALL_ID: tool_call.id, "name": tool_call.name, "args": dict(tool_call.args)})

                # ② 判定（纯函数，不阻塞）
                verdict = self._judge(tool_call)
                to_execute = verdict.call
                # ③ 要问人 → yield 请求，等消费者 send 回决定
                if verdict.needs_approval:
                    approved, by = yield from self._ask_approval(verdict, run_id, turn, run_span)
                    if approved:
                        self._audit("approval", to_execute, "allow", by, verdict.reason)
                    else:
                        reason = (f"{'人工拒绝' if by == ApprovalSource.HUMAN else '未获批准'}"
                                  f"（{verdict.reason}）")
                        self._audit("approval", to_execute, "deny", by, reason)
                        verdict = CallVerdict(action="deny", call=to_execute, reason=reason,
                                              source=DenySource.DENY_POLICY, risk=verdict.risk)
                # ④ 三条路：拒绝 / 委派（攒着，这一批跑完再并发跑）/ 其余（就地串行跑完）
                if verdict.is_denied:
                    result = denied_result(to_execute, verdict.reason, verdict.source, verdict.risk)
                elif self._is_delegation(to_execute):
                    if hold_from is None:
                        hold_from = index      # 被拒绝的委派走不到这里，所以它不会白攒一批
                    pending.append((index, to_execute, tool_span))
                    continue
                else:
                    result = execute_tool_call(to_execute, self.tools, exec_options)

                # ⑤ 结果落盘：**逐条 start → result**。攒到最后一起发就等于又变成"整批跑完再补发"
                #    （界面看不到逐条完成）；只有"这一批里有委派"时才不得不攒后面那几条。
                if hold_from is not None and index > hold_from:
                    held.append((index, result, tool_span))
                    continue
                yield self._tool_result_event(run_id=run_id, turn=turn, tool_span=tool_span, result=result)
                messages.append(result.message)
                if result.usage is not None:
                    # ★ 子代理花的也算这个 run 的：`result.usage` 是那棵子树的**全量**
                    #   （子代理自己的调用 ＋ 压缩 ＋ 它再委派的下级），所以直接累进去 ——
                    #   `RUN_END.usage` 就等于这个 run 烧掉的全部。想单独看委派那一块，
                    #   把本 run 里 `kind="subagent"` 的 `tool_result.usage` 加起来就是。
                    total_usage = total_usage + result.usage

            # ⑥ 委派并发跑完，再按**调用顺序**把攒下的结果发出去、回填历史。
            #    回填顺序必须与 `ai.tool_calls` 一致：那是"这一次请求"的形状，重放与下一轮
            #    发出去的历史都按它还原。`held` 非空时一定要发完 —— 它是被攒下的结果，
            #    不是可以省掉的东西。
            if pending:
                held.extend(self._run_delegations(pending, exec_options))
            for index, result, tool_span in sorted(held, key=lambda item: item[0]):
                yield self._tool_result_event(run_id=run_id, turn=turn, tool_span=tool_span, result=result)
                messages.append(result.message)
                if result.usage is not None:
                    total_usage = total_usage + result.usage

        yield Event(
            type=EventType.RUN_END, run_id=run_id, span=run_span,
            # `turn` 顺带记下"这次 loop 一共用了几轮"：run_end 是**最后**发射的那条，
            # 所以 `turn` 就是轮数（中途 break 也算，因为 break 之前 turn 已经涨过了）。
            # 这样遍历历史时**一个 RUN_END 就能读出**：总用量 ＋ 轮数 ＋ 结束原因。
            turn=turns_used,
            data={DataKey.STOP_REASON: stop_reason,
                  DataKey.DURATION_MS: int((time.monotonic() - started) * 1000),
                  DataKey.USAGE: total_usage}
        )

    # ---- 压缩与事件构造 ----
    def _compact(self, messages: Messages):
        """先试轻量剪切（零成本），压不动再试 AI 总结（要一次模型调用）。

        返回 ``None`` = 两档都压不动（调用方据此走"明确失败"）。
        """
        result = self.context.compact(messages, self.model.tools_schema)
        if result is not None:
            return result
        if self.summarizer is not None:
            return self.context.summarize(messages, self.summarizer)
        return None

    # ---- 委派：认出来、并发跑 ----
    def _is_delegation(self, call: ToolCall) -> bool:
        """这次调用是不是**委派**（`task`）。

        判据是注册表里那个工具的 `tool_type`，不是名字：`tool_type` 是建工具时定下来的
        （`subagent.py` 建 `task` 时给的就是 `SUBAGENT`），而名字有可能被外围换掉。
        查不到这个工具就当它不是委派 —— 那种调用本来也会被权限层按"名字不在已知清单里"拒掉。
        """
        tool = self.tools.get(call.name) if isinstance(self.tools, dict) else None
        return tool is not None and tool.tool_type == ToolType.SUBAGENT

    def _run_delegations(self, pending: list[tuple[int, ToolCall, str]],
                         exec_options: ExecOptions) -> list[tuple[int, ToolResult, str]]:
        """把这一批委派**并发**跑掉，返回 ``[(调用下标, 结果, span)]``（顺序不保证）。

        ★ **工作线程只做一件事：调 `execute_tool_call`。** 事件、`messages`、`total_usage`
        全留在主线程 —— 这个进程里只有主线程往会话文件里写东西，这条不破。

        ★ **运行期上下文要显式复制进线程。** `contextvars` 不跟着线程走，而 `task` 工具一进去
        第一件事就是 `get_run()` / `get_call()`（拿父 run 的 run_id、工作区、结果目录、记忆，
        以及这次调用的 call_id）。不复制的话它们在子线程里是空的，委派当场散架。每个分派
        各复制一份：同一个 `Context` 不能被两个线程同时进。
        """
        results: list[tuple[int, ToolResult, str]] = []
        workers = min(len(pending), MAX_CONCURRENT_DELEGATIONS)
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="delegate") as pool:
            futures = {}
            for index, call, span in pending:
                context = copy_context()
                futures[pool.submit(context.run, execute_tool_call, call, self.tools,
                                    exec_options)] = (index, span)
            for future in as_completed(futures):
                index, span = futures[future]
                results.append((index, future.result(), span))
        return results

    @staticmethod
    def _tool_result_event(*, run_id: str, turn: int, tool_span: str, result: ToolResult) -> Event:
        """`ToolResult` → `tool_result` 事件（含那几个**按需**才出现的键）。"""
        data: dict[str, Any] = {
            DataKey.DURATION_MS: result.duration_ms,
            DataKey.OUTCOME: result.outcome,
            DataKey.TOOL_TYPE: result.tool_type,
            DataKey.MESSAGE: result.message,
        }
        if result.truncated_from is not None:
            data[DataKey.TRUNCATED_FROM] = result.truncated_from
        if result.spilled_to is not None:
            data[DataKey.SPILLED_TO] = result.spilled_to
        if result.child_session is not None:
            data[DataKey.CHILD_SESSION] = result.child_session
        if result.usage is not None:
            data[DataKey.USAGE] = result.usage
        return Event(type=EventType.TOOL_RESULT, run_id=run_id, turn=turn, span=tool_span, data=data)

    # ---- 判定与审批（与执行分开：判在前、跑在后） ----
    def _judge(self, call: ToolCall) -> CallVerdict:
        """问判定链"这条能不能跑"。没配判定链 = 全部放行（单测 / 最简用法）。"""
        if self.judge is None:
            return CallVerdict(action="allow", call=call)
        verdict = self.judge(call)
        if verdict is None:                    # 判定链偷懒返回 None = 放行
            return CallVerdict(action="allow", call=call)
        self._audit("judge", verdict.call, verdict.action, verdict.source, verdict.reason)
        return verdict

    def _audit(self, kind: str, call: ToolCall, action: str, by: str, reason: str) -> None:
        """记账（可选）：内核只负责调，写不写由外层决定。"""
        if self.audit is not None:
            try:
                self.audit(kind=kind, call=call, action=action, by=by, reason=reason)
            except Exception:                  # noqa: BLE001  记账失败不该影响主流程
                pass

    def _ask_approval(self, verdict: CallVerdict, run_id: str, turn: int, run_span: str):
        """把"要不要批准"这件事问出去，返回 ``(approved, by)``。

        两条路：

        - **有人可问**（``approval_resolver`` 为 None）→ ``yield`` 一条 **live-only** 的
          ``approval_request`` 事件，消费者的 ``send(...)`` 回到这里。消费者可以 send：
          ``ApprovalDecision`` / ``True`` / ``False`` / **``None``（＝按默认值裁决）**。
          这是 TUI 那条路：弹窗、倒计时、人点按钮。
        - **没有人可问**（``approval_resolver`` 给了，例如子代理）→ 就地取一个决定，
          **不 yield**。子代理内部不审批（否则它的请求得从子会话冒到父消费者那里，
          那会把整条执行链变成生成器）。

        ``by`` 如实说明决定是谁做的；驱动方**不回答**时按"没有人可问"处理
        （::class:`ApprovalSource.NONE`）—— 与"没有通道就拒绝"同一条 fail-safe。
        """
        proposed = bool(self.propose(verdict.call)) if self.propose is not None else False
        asked_ts = now_ms()
        if self.approval_resolver is not None:
            request = self._approval_request(verdict, proposed, asked_ts)
            decision = self.approval_resolver(request)
            return self._interpret(decision, proposed)

        request = self._approval_request(verdict, proposed, asked_ts)
        answer = yield Event(
            type=EventType.APPROVAL_REQUEST, run_id=run_id, turn=turn, span=run_span,
            data={"request": request})
        # 消费者 send 回来的值 → 一个决定
        return self._interpret(answer, proposed)

    @staticmethod
    def _approval_request(verdict: CallVerdict, proposed: bool, asked_ts: int) -> ApprovalRequest:
        timeout_s = 60.0                      # 弹窗自己该等多久（驱动方可自行缩短/忽略）
        return ApprovalRequest(
            call=verdict.call, reason=verdict.reason, risk=verdict.risk,
            invocation=f"{verdict.call.name}({verdict.call.id})",
            asked_ts=asked_ts, timeout_s=timeout_s,
            expires_ts=asked_ts + int(timeout_s * 1000), proposed=proposed)

    @staticmethod
    def _interpret(answer, proposed: bool) -> tuple[bool, str]:
        """把驱动方 send 回来的东西解释成 ``(approved, by)``。

        宽松入口（TUI 不必 import 词表也能用）：``ApprovalDecision`` / ``bool`` /
        ``(决定, by)`` 元组 / ``None``（＝按默认值，``by="none"``）。
        """
        if answer is None:
            return proposed, ApprovalSource.NONE
        if isinstance(answer, ApprovalDecision):
            return answer == ApprovalDecision.APPROVED, ApprovalSource.HUMAN
        if isinstance(answer, bool):
            return answer, ApprovalSource.HUMAN
        if isinstance(answer, tuple) and len(answer) == 2:
            decision, by = answer
            approved = (decision == ApprovalDecision.APPROVED if not isinstance(decision, bool)
                        else decision)
            return bool(approved), str(by)
        return proposed, ApprovalSource.NONE