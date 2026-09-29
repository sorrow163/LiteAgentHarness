# -*- coding: utf-8 -*-
"""沙箱家族：策略、`confine` 接缝、各平台后端。

## 现在有什么 / 没有什么

这里现在有**两件东西**，它们是一层接一层的：

- `restricted_token.py` —— **机制**：写受限令牌（write-restricted token）。造令牌、挂 ACE、
  起进程、收输出，只认 `argv`。**真机已验证**：`tests/test_sandbox_a.py` 14/14，
  6 条未知数零跳过。
- `bash.py` —— **接线**：工具 `sandbox_bash`（config ＋ 包装 ＋ 工具）。把"一条命令字符串"
  翻成 argv、把结果翻成给模型看的一段字、把可调项收成 `SandboxConfig`。受众是接线的人
  与**模型**（工具说明就是它的接口文档）。

**只限制写，读不受限** —— 于是工作区里那个 `.venv`、祖先目录、向上找配置、NUL 设备
全都不需要额外授权，工具链行为与裸机一致。代价是"区外读得到"（见 `restricted_token.py`）。

它**共享宿主的内核与文件系统**，所以工作区里那个 `.venv` 就是它自己的 `.venv` ——
这正是它相对于"容器"路线的全部意义：容器换了一个 OS，Windows 的 venv 在 Linux 里用不了，
而宿主内沙箱不换 OS。

**被否决的两档**（记录在别处，别在这里重开）：

- **WSL + Docker** —— 换 OS，就是上面那条理由。
- **AppContainer** —— 边界更硬（读写都拒），但"打不开 NUL 设备"修不掉：pytest 的 capture
  与 logging 插件都默认开 `nul`，于是跑 pytest 必须带 `-s -p no:logging`。
  那个模块（`appcontainer.py`）**已按决定删除**，实测记录与抢救出来的坑在
  `docs/sandbox-appcontainer-postmortem.md`；当时的探针还留在 `temp/probe_appcontainer_*.py`
  等文件里，但它们 import 不到那个模块了，只能当原始记录看。

**docker 那一档已删**：它原来是一个兄弟模块 `src/harness/docker_sandbox.py`（"宿主 → WSL →
容器"），比这个包先存在。接线换到本包之后它就没有调用方了，所以连同 `tests/test_sandbox.py`
一起删掉 —— 留着就是"让人以为沙箱还可能是容器"的死代码。它的否决理由在
`docs/GOAL.md` 第十一节。

## 为什么这里没有"转发旧名字"的兼容层

这个 `__init__.py` 一度只做一件事：把 docker 那一档的旧名字转发一遍，好让
`from src.harness.sandbox import Sandbox` 之类的旧写法一行都不改。后来那些写法都直接改指
`docker_sandbox` 了，转发层就只剩一个消费者 —— 而且它有害：它让
`src.harness.sandbox.CommandResult` 读起来像"沙箱家族的公共类型"，实际却指向 docker 后端。
那比没有更容易看错，所以拆掉了。

（它本来也不该长期存在：那不是兼容层，是把耦合藏起来。）

## 约定

- **一个后端一个模块**，文件名就是它的方言名（`restricted_token.py`，将来的 `posix.py` 之类）。
  **这个包不做 re-export** —— 消费者照直写
  `from src.harness.sandbox import restricted_token`，谁是谁一目了然。
- 公共词汇（策略、模式、结果记录、拒绝标记）将来放这一包的**顶层模块**，各后端共用。
  `ConfinedRun` 的字段与**历史上** docker 那一档的 `CommandResult` 是对齐的（那是刻意的：
  接线的调用方换后端时不用改读数的地方），但**不预先抽象** —— 空的抽象比没有抽象更难改。
"""
