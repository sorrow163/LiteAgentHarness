# -*- coding: utf-8 -*-
"""`Harness.run_stream()` 与界面之间的桥：**一个 run 一个工作线程**。

## 为什么必须换线程

`Harness.run_stream()` 是生成器，而审批是**双向驱动**的：内核 ``yield`` 一条
``approval_request``，驱动方用 ``gen.send(决定)`` 把答案送回那个挂起点
（`src/core/loop.py` 的 `_ask_approval`）。而生成器**不能被两个线程同时推进** ——
一个正在执行的生成器，再从别的线程 ``send`` 进去会抛
``ValueError: generator already executing``。所以 ``next()`` 与 ``send()`` 必须落在
**同一个线程**里。

结论：整个 run 都放进工作线程，界面线程只通过两个方向说话 ——
**事件出去**（回调）、**决定进来**（队列）。

## 谁负责"超时"

`ApprovalRequest.timeout_s` / ``expires_ts`` 是内核给的"弹窗自己该等多久"。等超时的
**执行**放在本模块，不放在界面里：界面只画倒计时，到点了由本线程按默认值继续往下走。
理由是"没有人回答"这件事必须有一条确定的路走完 —— 否则界面一崩、或者用户把终端最小化
忘了，run 就永远挂在一个没人答的审批上。

超时送回去的答案**不是** ``None``：``None`` 在 `loop._interpret` 里等于"没有人可问"
（``ApprovalSource.NONE``），而这里的事实是"问过人了、人没答"（``ApprovalSource.TIMEOUT``）。
这两件事在记账里的意义相反（前者衡量配置有多危险，后者衡量界的响应），所以这里送
``(proposed, ApprovalSource.TIMEOUT)`` 这个二元组 —— `_interpret` 明确支持这种宽松入口。

## 谁负责"界面还活着"

本模块**不认识界面**：五个回调都是普通可调用对象，随便换成打印、测试里的列表收集都行。
`tui.app` 把回调包一层 ``call_from_thread`` 送回界面线程，那是它的事。
"""
from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import Any

from src.core.events import ApprovalSource, Event, EventType, now_ms

#: 等答案时比 ``expires_ts`` 多留的一点余量（秒）。
#: 界面按 ``expires_ts`` 画倒计时，两边**同一个截止时刻**；留这条缝是为了"人正好在最后一刻
#: 按下按钮"时不被判成超时 —— 早收十几毫秒的拒绝，比迟收十几毫秒的批准代价大。
ANSWER_GRACE_S = 0.5


class RunBridge:
    """驱动一次 run：事件出去、决定进来。

    用法::

        bridge = RunBridge(harness, on_event=..., on_approval=..., on_answered=...,
                           on_error=..., on_finish=...)
        bridge.start("帮我把测试跑通")
        bridge.answer(index, ApprovalDecision.APPROVED)   # 界面按了"批准"
    """

    def __init__(
        self,
        harness: Any,
        *,
        on_event: Callable[[Event], None],
        on_approval: Callable[[Any, int], None],
        on_answered: Callable[[int, str], None],
        on_error: Callable[[BaseException], None],
        on_finish: Callable[[], None],
    ) -> None:
        self.harness = harness
        self._on_event = on_event
        self._on_approval = on_approval
        self._on_answered = on_answered
        self._on_error = on_error
        self._on_finish = on_finish
        self._decisions: queue.Queue[tuple[int, Any]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        #: 每条审批请求的序号。界面答完把序号带回来，用来**丢掉过期的决定**
        #: （超时之后界面才把答案送回来、或者上一轮的弹窗迟了一拍）。
        self._next_index = 0
        self._current_index = -1

    # ---- 状态 ----
    @property
    def running(self) -> bool:
        """这一轮还在跑吗。"""
        return self._running.is_set()

    # ---- 界面的三个动作 ----
    def start(self, prompt: str, *, session_id: str | None = None) -> None:
        """起一轮 run。**上一轮没结束就报错**：同一时刻只允许一个 run。

        多个 run 并行不在能力面里（工具执行本身是串行的，见 `docs/contracts/tools.md`），
        硬开两条只会让事件流的时间线互相穿插、`interrupt()` 也说不清打断的是谁。
        """
        if self.running:
            raise RuntimeError("上一轮还没结束 —— 同一时刻只允许一个 run")
        self._running.set()
        self._thread = threading.Thread(
            target=self._drive, args=(prompt, session_id), name="harness-run", daemon=True)
        self._thread.start()

    def answer(self, index: int, decision: Any) -> bool:
        """界面 → 工作线程。返回"这条决定有没有被收下"。

        序号对不上说明那条请求已经翻篇（超时了、或者界面迟了一拍），**丢掉**：
        把过期的决定塞进队列，会被**下一次**审批当成答案取走 —— 用户会看到
        "我明明点了拒绝，它却批了"。
        """
        if index != self._current_index:
            return False
        self._decisions.put((index, decision))
        return True

    def interrupt(self) -> None:
        """请求停止当前 run（协作式：内核在下一个消息边界生效）。"""
        self.harness.interrupt()

    def inject(self, text: str) -> None:
        """中途插话（steering）：内核在下一轮模型调用前取走，并落成一条 ``user_message``。"""
        self.harness.inject(text)

    # ---- 工作线程 ----
    def _drive(self, prompt: str, session_id: str | None) -> None:
        try:
            gen = self.harness.run_stream(prompt, session_id)
            answer: Any = None
            while True:
                try:
                    # ★ 有答案就 send、没有就 next：send 的值会经 `run_stream` 里的
                    #   `inner.send(answer)` 一路送到循环的审批挂起点。
                    event = gen.send(answer) if answer is not None else next(gen)
                except StopIteration:
                    return
                self._on_event(event)
                answer = None
                if event.type == EventType.APPROVAL_REQUEST:
                    answer = self._wait_for_answer(event.data["request"])
        except BaseException as exc:                 # noqa: BLE001
            # 钩子拒绝输入、会话打不开、模型构造失败一类**发生在事件流之外**的错：
            # 它们没有对应的事件，只能从异常这条路交出去（`ERROR` 事件是另一回事）。
            self._on_error(exc)
        finally:
            self._current_index = -1
            self._running.clear()
            self._on_finish()

    def _wait_for_answer(self, request: Any) -> Any:
        """把一条审批请求交出去，然后**等**答案（或等到超时）。"""
        # 先丢掉上一轮的残渣（见 `answer` 的说明），再认下这一条的序号 —— 顺序不能反，
        # 反了会把"刚刚答的"当成残渣丢掉。
        while True:
            try:
                self._decisions.get_nowait()
            except queue.Empty:
                break

        index = self._next_index
        self._next_index += 1
        self._current_index = index
        self._on_approval(request, index)

        wait_s = max(0.05, (request.expires_ts - now_ms()) / 1000 + ANSWER_GRACE_S)
        try:
            _, decision = self._decisions.get(timeout=wait_s)
        except queue.Empty:
            self._current_index = -1
            self._on_answered(index, "timeout")
            return (bool(request.proposed), ApprovalSource.TIMEOUT)

        self._current_index = -1
        self._on_answered(index, "human")
        return decision
