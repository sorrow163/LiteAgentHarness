# -*- coding: utf-8 -*-
"""终端界面（TUI）：`Harness` 的**消费者**。

## 这一层的位置

`src/core/` 是内核（事件流 ＋ agent 循环），`src/harness/` 是装配层（会话、能力面、
控制面、权限）。**本层不往上面任何一层加机制** —— 它只做三件事：把事件流画出来、
把人的决定送回去、把状态栏拼出来。需要的东西都在接口上：

| 界面要做的 | 用的接口 | 在哪 |
|---|---|---|
| 跑一轮并拿到流 | `Harness.run_stream(prompt)` | `src/harness/harness.py` |
| 回答审批 | `gen.send(决定)` | `src/core/loop.py` `_ask_approval` |
| 打断 | `Harness.interrupt()` | 同上 |
| 中途插话 | `Harness.inject(text)` | 同上 |
| 换权限模式 | `policy.set_mode()` / `policy.modes` | `src/harness/permissions.py` |
| 会话列表 / 恢复 / 回放 | `list_session_summaries()` / `open(id)` / `Session.events()` | `src/harness/harness.py` · `session.py` |
| 子会话树 | `child_sessions(id)` / `SessionStore.events_of(id)` | `src/harness/harness.py` · `session.py` |

## 模块

- :mod:`tui.render` —— 事件 → 一行行文本。**纯函数、不 import 界面框架**，所以能单测。
- :mod:`tui.sessions` —— 会话文件 → 会话树（四态状态、用量）。**同样是纯的**，所以不启动
  终端也能测。看子会话是**读文件**，不是"打开会话"（后者会往那条委派记录里写）。
- :mod:`tui.bridge` —— 把一个 run 放进工作线程驱动，事件出去、决定进来。
- :mod:`tui.widgets` —— 控件：时间线、状态栏、审批弹窗、会话树、只读的子会话视图。
- :mod:`tui.app` —— 把它们装成一个应用（布局、按键、事件泵）。
- :mod:`tui.console` —— 启动前把控制台切成 UTF-8（否则画不出那些字符）。
- :mod:`tui.__main__` —— 命令行：读配置（三层优先级）、**启动时校验 MCP 配置**、装配 `Harness`、
  开会话、跑界面、退出时收尾。进程级的生命周期（会话什么时候开、什么时候关）都在这里，
  界面只负责画和驱动。

## 依赖

界面框架（`textual`）是**可选依赖**：`pip install -e ".[tui]"`。内核与装配层
一行都不依赖它 —— `tui/` 不 import 时，整个项目照常跑测试与评估。
"""
