# -*- coding: utf-8 -*-
"""启动终端界面。

用法（在仓库根跑，与 `python -m src.eval.run` 一致）::

    python -m tui                      # 当前目录当工作区，新建会话
    python -m tui D:\\some\\project      # 指定工作区
    python -m tui --list               # 只列会话（当前工作区的），不进界面
    python -m tui --session 20260928-153012-ab12cd   # 接着某条会话聊
    python -m tui --show-capabilities   # 只打印能力面（工具／技能／子代理／沙箱／MCP／模型），不进界面
    python -m tui --show-tree 20260928-153012-ab12cd # 只打印会话树（子代理的子子孙孙），不进界面
    python -m tui --show-sub 20260928-153012-ab12cd.sub-1   # 只打印一条会话的时间线，不进界面

## 配置从哪来

**命令行 > `config.yml` > 代码里的默认值**（见 `src/config.py`）。长期配置写进
`config.yml`（默认就是项目本目录那份，`--config` 可以指到别处），命令行只用来临时盖一次。
所以这个文件里剩下的参数都是**覆盖项**：它们的默认值是 `None`（＝"没说"），
没说就用配置文件里的值 —— 而不是把默认值再抄一份在这儿（抄一份就是第二个真相）。

**只有 `model.base_url` 与 `model.api_key` 是必填的**（在配置文件里），少了当场报错。

## 能力面是"配了才有"的

技能、子代理、沙箱、MCP、摘要模型**各有各的开关**，不开就没有：没有技能目录就没有 `read_skill`、
没注册子代理就没有 `task`、不开沙箱就没有 `sandbox_bash`、`mcp.config` 留空就没有 MCP 工具、
不给摘要模型就只剩"清掉旧工具结果"这一档压缩。它们既能在 `config.yml` 里配，也能用命令行临时
盖一层，并用 `--show-capabilities` 让人**不打开会话、不进界面**就核对一遍。

**MCP 那一档在启动时是硬失败**：`mcp.config` 一旦指向某个文件，那份文件就会被整份校验
（在不在、能不能解析、有没有 `mcpServers`、每项的 `command`/`args`/`env`/`timeout_s` 对不对、
命令找不找得到），任一条不过就**打警告并停止启动**。别的能力坏了是"少一个工具"，
MCP 坏了是"你以为有一批工具、其实一个都没有"，那要到模型真去调它时才暴露。

## 会话的开关归这里管

进程级生命周期：界面拿到的是一个已经打开的 `Session`，退出时由这里 `close()` ——
界面只负责画和驱动，不管"会话什么时候开、什么时候关"。
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.config import (
    Config, ConfigError, build_mcp_servers, build_model, config_path_of, default_config,
    describe_model, load_config)
from src.core.models import FakeModel
from src.core.toolkit import make_coding_tools
from src.harness.harness import DEFAULT_SESSION_ROOT, Harness, HarnessConfig
from src.harness.permissions import PermissionPolicy
from src.harness.sandbox.bash import SandboxConfig
from src.harness.skills import SKILL_FILE
from src.harness.subagent import SubagentDef

from tui import render, sessions
from tui.console import ensure_utf8_console

#: 会话 id 的形状 —— 与 `SessionStore.create` 造 id 的写法一致（``日期-时间-6 位十六进制``）。
SESSION_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")

#: 没显式给技能目录时去这里找（**工作区内**的约定目录）。
#: `.harness/` 是仓库里已有的位置约定：工具结果转存就落 `<工作区>/.harness/results/`
#: （见 `docs/contracts/tools.md`）。技能跟着它走，用户不必记第二套路径。
DEFAULT_SKILLS_SUBDIR = (".harness", "skills")


def _looks_like_session_id(value: str) -> bool:
    """这个位置参数长得像会话 id 吗（只看最后一段名字，路径分隔符无所谓）。"""
    return bool(SESSION_ID_RE.match(os.path.basename(os.path.normpath(value))))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tui", description="LiteAgentHarness 的终端界面")
    parser.add_argument("workspace", nargs="?", default=".",
                        help="工作区（默认当前目录）—— 工具按它解析路径。"
                             "**刻意不进配置文件**：每次跑都可能不同")
    parser.add_argument("--config", default=None,
                        help="配置文件路径（默认 <仓库根>/config.yml）")

    # ---- 一次性动作 ----
    parser.add_argument("--list", action="store_true",
                        help="只列出当前工作区的会话，不进界面")
    parser.add_argument("--session", help="接着这条会话聊（id 见 --list）")
    parser.add_argument("--session-root", default=None,
                        help="会话文件目录（默认 <仓库根>/sessions）")
    parser.add_argument("--show-capabilities", action="store_true",
                        help="只打印能力面（工具／技能／子代理／沙箱／MCP／模型），不进界面")
    parser.add_argument("--show-tree", metavar="会话id", default=None,
                        help="只打印会话树（以这条会话为根的子代理子子孙孙），不进界面")
    parser.add_argument("--show-sub", metavar="会话id", default=None,
                        help="只打印一条会话的时间线（纯文本），不进界面。"
                             "**只读**：不会往那条会话里写任何东西")

    # ---- 覆盖项：默认 None ＝"没说"，那就用配置文件里的值 ----
    parser.add_argument("--model", help="覆盖配置里的模型名")
    parser.add_argument("--provider", help="覆盖配置里的 provider（只影响元信息与思考模式判断）")
    parser.add_argument("--mode", default=None, choices=PermissionPolicy().modes,
                        help="覆盖配置里的权限模式")
    parser.add_argument("--max-turns", type=int, default=None,
                        help="覆盖配置里的一轮最大模型调用次数")
    parser.add_argument("--sandbox", action=argparse.BooleanOptionalAction, default=None,
                        help="覆盖配置里的沙箱开关（--sandbox / --no-sandbox）")
    parser.add_argument("--skills-dir", default=None,
                        help=f"覆盖配置里的技能目录（扫 <目录>/*/{SKILL_FILE}）；"
                             f"--skills-dir '' 表示不要技能")
    parser.add_argument("--subagent", action="append", default=None,
                        metavar="名字|描述|system prompt",
                        help="覆盖配置里的子代理（可重复）。描述是写给**主模型**看的，"
                             "它靠这段决定什么任务派给谁")
    parser.add_argument("--summarizer-model", default=None,
                        help="覆盖配置里摘要用的模型名（默认与主模型相同）")
    parser.add_argument("--no-summarizer", action="store_true",
                        help="不要摘要模型：上下文只剩清旧工具结果那一档可压")
    parser.add_argument("--fold-chars", type=int, default=None,
                        help="覆盖配置里的折叠阈值（思考块超过这么多字符就折起来）")
    return parser


def parse_subagents(specs: Sequence[str]) -> list[SubagentDef]:
    """``"名字|描述|system prompt"`` → `SubagentDef` 清单（命令行那条路）。

    **用竖线分段，不用 JSON**：Windows 上 PowerShell 会把原生命令参数里的双引号吃掉
    —— 这个仓库真踩过（`--mounts '{"json": 1}'` 到程序里成了 `{json: 1}`），
    所以凡是命令行传结构化数据，一律避开引号。配置文件那条路没这个问题，那边写 YAML 列表。
    """
    out: list[SubagentDef] = []
    for spec in specs:
        parts = [part.strip() for part in spec.split("|")]
        if len(parts) != 3 or not all(parts):
            raise ValueError(
                f"--subagent 要写成 '名字|描述|system prompt' 三段（竖线分隔），收到：{spec!r}")
        out.append(SubagentDef(name=parts[0], description=parts[1], system_prompt=parts[2]))
    return _reject_duplicate_names(out)


def build_subagents(args: argparse.Namespace, cfg: Config, workspace: str) -> list[SubagentDef]:
    """配置里的 ＋ 命令行给的 → 交给 `HarnessConfig` 的子代理清单。

    ★ **没写工具的，默认给父代理那套编码工具**：`make_task_tool` 只把 ``spec.tools`` 交给
    子代理，而一个工具都没有的子代理**连文件都读不了**（它只能"说话"）—— 那样委派出去毫无意义。
    所以默认是"同一套工作区工具"，而不是"空"。（沙箱那一档不给子代理：它是父代理的额外能力。）
    """
    entries = (parse_subagents(args.subagent) if args.subagent else
               [SubagentDef(name=s.name, description=s.description, system_prompt=s.system_prompt)
                for s in cfg.subagents])
    coding = make_coding_tools(workspace)
    return [replace(sub, tools=list(sub.tools) or coding) for sub in entries]


def _reject_duplicate_names(subs: list[SubagentDef]) -> list[SubagentDef]:
    """重名在 `make_task_tool` 里是**静默覆盖**（它按名字建 dict），所以在入口拦下。"""
    seen: set[str] = set()
    for sub in subs:
        if sub.name in seen:
            raise ValueError(f"子代理名字重复：{sub.name!r}（后者会静默盖掉前者）")
        seen.add(sub.name)
    return subs


def resolve_skills_dir(args: argparse.Namespace, cfg: Config, workspace: str) -> str | None:
    """技能目录：命令行 > 配置文件 > 工作区里有没有 ``.harness/skills``。

    **不存在就当没有**（`load_skills` 对不存在的目录返回空清单）：不硬塞一个空目录，
    也不因为"没建技能目录"而报错 —— 大多数工作区本来就没有技能。
    """
    if args.skills_dir is not None:               # 显式给了就用它（空串 = 明确不要技能）
        return args.skills_dir or None
    if cfg.skills_dir:
        return cfg.skills_dir
    candidate = Path(workspace).joinpath(*DEFAULT_SKILLS_SUBDIR)
    return str(candidate) if candidate.is_dir() else None


def print_sessions(harness: Harness) -> None:
    """列会话：**进界面前先知道有哪些 id 可以接**（`--session` 要用它）。"""
    summaries = harness.list_session_summaries()
    if not summaries:
        print(f"当前工作区还没有会话：{harness.cfg.workspace}")
        return
    print(f"当前工作区的会话（新 → 旧）：{harness.cfg.workspace}")
    for item in summaries:
        state = "已结束" if item["closed"] else "未结束"
        print(f"  {item['id']}  {item['model'] or '?':<24} "
              f"{item['message_count']:>4} 条消息  {state}")


# ---------------------------------------------------------------------------
# 只读出口：不进界面也能看子会话（`--show-tree` / `--show-sub`）
# ---------------------------------------------------------------------------
def print_tree(harness: Harness, root_id: str, *, fold_chars: int = render.FOLD_CHARS) -> int:
    """把会话树打成纯文本，如::

        20260929-143211-80e606 · 主会话 · 15 轮 · 96 步 · 2.5M tok
          ◆ sub-1 · explorer · 完成 · 2 轮 · 7 步 · 4.2k tok · “看看沙箱怎么做的”
            ○ sub-1.sub-1 · reader · 未收尾 · 1 轮 · 2 步 · 900 tok · “读一下那个文件”

    为什么要有这条路：**子会话树的数据全在文件里**，界面只是把它画出来。把它也做成一个
    纯文本出口，"这棵树对不对"就不必靠盯着屏幕看 —— 人和测试用的是同一个东西。

    折叠阈值取 ``--fold-chars``，没给就用代码默认值：这条路和 `--list` 一样**不读
    `config.yml`**（"看看会话文件里写了什么"跟 key、跟模型都没关系），所以配置里那个
    ``ui.fold_chars`` 在这儿不生效。
    """
    if not _has_session(harness, root_id):
        print(f"没有这条会话：{root_id}", file=sys.stderr)
        return 2
    node = sessions.build_tree(harness.store, root_id)
    for depth, item in _flatten(node):
        print("  " * depth + render.strip_markup(sessions.node_label(item)))
    return 0


def print_timeline(harness: Harness, session_id: str,
                   *, fold_chars: int = render.FOLD_CHARS) -> int:
    """把一条会话的时间线打成纯文本（**只读**，对子会话和主会话都能用）。

    跟着**屏幕的默认状态**走：工具正文折着、超长的思考只给开头一段（见
    :func:`tui.render.block_text`）—— 不然一份时间线会被工具结果淹掉。

    和 `--show-tree` 一样**不读 `config.yml`**，折叠阈值只认 ``--fold-chars``。
    """
    node = sessions.read_one(harness.store, session_id)
    try:
        events = harness.store.events_of(session_id)
    except OSError:
        print(f"没有这条会话：{session_id}", file=sys.stderr)
        return 2

    print(render.strip_markup(sessions.header_line(node)))
    prompt = sessions.header_prompt(node)
    if prompt:
        print(render.strip_markup(prompt))
    print()
    events = events if node.main else sessions.strip_bracket(events)
    for block in render.replay(events, fold_chars=fold_chars).blocks:
        print(render.block_text(block))
        print()
    return 0


def _has_session(harness: Harness, session_id: str) -> bool:
    """这条会话在不在。

    ★ **不能用 ``store.list()`` 判**：那个走 ``iter_heads()`` 的默认参数（``parent=-1``），
    **只列主控** —— 拿它去查一条子会话，得到的答案是"不存在"，而文件明明就在那儿。
    """
    try:
        harness.store.events_of(session_id)
    except OSError:
        return False
    return True


def _flatten(node: Any, depth: int = 0) -> list[tuple[int, Any]]:
    out = [(depth, node)]
    for child in node.children:
        out.extend(_flatten(child, depth + 1))
    return out


def make_harness(args: argparse.Namespace, cfg: Config, workspace: str, model: Any, *,
                 coding_tools: bool = True, summarizer: Any = None,
                 subagents: Sequence[SubagentDef] = (),
                 mcp_servers: Sequence[Any] = ()) -> Harness:
    """按"命令行 > 配置文件 > 代码默认值"装配一个 `Harness`。

    ``coding_tools=False`` 是给 `--list` 用的：那套工具**一装配就会 `mkdir(workspace)`**
    （`make_coding_tools` 要有一个根才能解析路径），而"看有哪些会话"不该在工作区上留下
    任何东西 —— 尤其是那个工作区可能根本不该存在（路径敲错时，`--list` 会先替你把它建出来）。
    """
    return Harness(HarnessConfig(
        model=model,
        # workspace 与 session_root **不进配置文件**：前者每次跑都可能变，后者是固定的。
        workspace=workspace,
        session_root=args.session_root or DEFAULT_SESSION_ROOT,
        permission_mode=args.mode or cfg.permission_mode,
        max_turns=args.max_turns or cfg.max_turns,
        max_context_tokens=cfg.max_context_tokens,
        coding_tools=coding_tools,
        skills_dir=resolve_skills_dir(args, cfg, workspace),
        subagents=list(subagents),
        summarizer=summarizer,
        mcp_servers=list(mcp_servers),
        sandbox=(SandboxConfig(workspace=workspace)
                 if (cfg.sandbox if args.sandbox is None else args.sandbox) else None),
    ))


def print_capabilities(harness: Harness, cfg: Config, *, model_shown: str,
                       summarizer_shown: str, mcp_servers: Sequence[Any] = (),
                       note: str = "") -> None:
    """把**能力面**打出来。开关的由来很长，一句话：能配的东西就得能核对。

    工具清单读的是 ``harness.policy.known``（装配时回填的"当前实际有哪些工具"），
    所以它不会和真正交给模型的那份走散。**沙箱那一档例外**：`sandbox_bash` 是在
    `open()` 里懒装配的（挂 ACE 要遍历工作区树，能省就省），这里不打开会话，所以单独说明。
    """
    tools = sorted(harness.policy.known)
    print(f"配置文件：{cfg.path}")
    print(f"工作区：{harness.cfg.workspace}")
    print(f"权限模式：{harness.cfg.permission_mode}"
          f"（可选：{' / '.join(PermissionPolicy().modes)}）")
    print(f"模型：{model_shown}")
    if note:
        print(f"  （{note}）")
    print(f"工具（{len(tools)}）：{'、'.join(tools) if tools else '(无)'}")
    print(f"摘要模型：{summarizer_shown}")

    skills = harness.skills
    where = harness.cfg.skills_dir or f"未指定（也没找到 <工作区>/{'/'.join(DEFAULT_SKILLS_SUBDIR)}）"
    print(f"技能（{len(skills)}）：{'、'.join(s.name for s in skills) if skills else '无'}"
          f"　← {where}")

    subs = list(harness.cfg.subagents)
    print(f"子代理（{len(subs)}）："
          + ("、".join(f"{s.name}（{s.description}）" for s in subs) if subs else "无"))

    print("沙箱：开（多一个 sandbox_bash；宿主内写受限令牌，只挡写、不挡读）" if harness.cfg.sandbox
          else "沙箱：关（配置文件里 capabilities.sandbox 或 --sandbox 打开）")
    if harness.cfg.sandbox:
        print("  （沙箱那一档要打开会话才装配，所以它不在上面那份工具清单里）")

    # MCP 同样不在上面那份工具清单里：真实名字要等 server 起来、问完 tools/list 才知道
    # （`<server>__<tool>` 这个前缀也是那时拼的）。这里报的是**配置里声明了什么**。
    if mcp_servers:
        print(f"MCP（{len(mcp_servers)}）："
              + "、".join(f"{s.name}（{' '.join(s.command)}）" for s in mcp_servers))
        print(f"  ← {cfg.mcp_config}")
        print("  （MCP server 要打开会话才启动 —— 要起子进程、要握手，"
              "所以它的工具要那时才进工具清单）")
    else:
        print("MCP：无（配置文件里 mcp.config 留空就是不用；填了就在启动时整份校验）")


def announce_new_workspace(workspace: str) -> None:
    """工作区不存在就**明说一句**。

    工具层会顺手把它建出来（`src/core/toolkit.py` 里那句 `mkdir`），不说的话，用户只会
    事后在磁盘上发现一个意料之外的目录。这个坑真被踩过：把会话 id 当工作区传进去，
    工作区里就长出一个以会话 id 为名的空文件夹，而且当时谁都不知道是那条命令干的。
    """
    if not os.path.isdir(workspace):
        print(f"工作区不存在，将新建：{workspace}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = str(Path(args.workspace).resolve())
    # ★ **一进来就切 UTF-8**，而不是等到起界面：`--list` 与报错也是这个程序打出去的，
    #   而中文在 GBK 管道里就是乱码（仓库里所有落盘的东西也一律 UTF-8）。
    restore_console = ensure_utf8_console()
    try:
        return _run(args, workspace)
    finally:
        restore_console()


def _run(args: argparse.Namespace, workspace: str) -> int:
    if _looks_like_session_id(args.workspace) and not os.path.isdir(workspace):
        # 这个坑真的被人踩过：状态栏上显示的是**会话 id**，而它长得和"一个项目目录名"
        # 没区别，于是"接着上次聊"很容易写成 `python -m tui 20260929-132804-9749a9`。
        # 后果不只是报错，而是**悄悄建出一个空目录**当工作区（工具层的 `make_coding_tools`
        # 会 `mkdir`），下一次再看见它就分不清那到底是项目还是垃圾。
        # 宁可在这里停下说清楚 —— 接着某条会话的入口是 `--session`。
        print(
            f"{args.workspace!r} 看着像**会话 id**，不像工作区。\n"
            f"  要接着那条会话聊：python -m tui --session {args.workspace} <工作区>\n"
            f"  要把它当工作区用：先把目录建出来（现在它不存在，会被当成新工作区）。",
            file=sys.stderr)
        return 2

    if args.list:
        # ★ 列会话**不读配置、也不造模型**：它只读会话目录。要配置文件里配齐 key
        #   才能看"我上次那条会话叫什么"，是说不通的。
        #   也**不装配编码工具**：那样连工作区都不会被建出来（见 `make_harness`）。
        print_sessions(make_harness(args, default_config(), workspace, FakeModel(),
                                    coding_tools=False))
        return 0

    if args.show_tree or args.show_sub:
        # 同样是**纯读**的出口：不读配置、不造模型、不建工作区（`coding_tools=False`）。
        # 这几样都跟"会话文件里写了什么"无关 —— 而会话树的数据**全在文件里**。
        harness = make_harness(args, default_config(), workspace, FakeModel(), coding_tools=False)
        try:
            if args.show_tree:
                return print_tree(harness, args.show_tree,
                                  fold_chars=args.fold_chars or render.FOLD_CHARS)
            return print_timeline(harness, args.show_sub,
                                  fold_chars=args.fold_chars or render.FOLD_CHARS)
        finally:
            harness.close()

    config_note = ""
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        if not args.show_capabilities:
            print(f"配置有问题：{exc}", file=sys.stderr)
            return 2
        # 能力面**跟 base_url / api_key 没关系**（工具、技能、子代理、沙箱都由配置与代码默认值
        # 决定），所以核对能力时不该因为没配 key 就跑不了。列完就说清用的是默认值。
        cfg = default_config(config_path_of(args.config))
        config_note = f"配置没读成（{exc}）—— 下面全是代码里的默认值"

    # ---- MCP：**启动时就把那份配置整份校验完，坏了就停下** ----
    # 这是**硬失败**，连 `--show-capabilities` 也不放行（那边只对模型配置宽容）：
    # 配了 MCP 就意味着这套能力面里该有那些工具，而"少了一批工具却照常起界面"的后果，
    # 要到模型真的去调它的时候才暴露 —— 那时人已经在一个残缺的环境里干了一会儿活了。
    # 所以：宁可在启动的那一秒报清楚哪一行不对。
    # ★ 位置也很要紧：**排在所有会碰工作区的动作之前**（`build_subagents` 一装配工具就会
    #   `mkdir(workspace)`）。决定不启动，就一个目录都不该被建出来。
    # 注意这里**只校验、不连 server**：起子进程、握手是 `harness.open()` 的事（见 build_mcp_servers）。
    try:
        mcp_servers = build_mcp_servers(cfg)
    except ConfigError as exc:
        print(f"MCP 配置有问题：{exc}", file=sys.stderr)
        return 2

    try:
        subagents = build_subagents(args, cfg, workspace)
    except ValueError as exc:                      # --subagent 写错了
        print(str(exc), file=sys.stderr)
        return 2

    # ---- 模型：命令行 > 配置文件 ----
    resolved_model_name = args.model or cfg.model.name
    model_shown = ""                               # 下面按"造没造出来"填
    model: Any = FakeModel()
    summarizer: Any = None
    if args.no_summarizer:
        summarizer_shown = "关（--no-summarizer）：上下文只剩清旧工具结果那一档可压"
    else:
        summarizer_name = (args.summarizer_model or cfg.summarizer_model or resolved_model_name)
        source = "配置里指定" if (args.summarizer_model or cfg.summarizer_model) else "与主模型相同"
        summarizer_shown = f"{summarizer_name}（{source}，另起一个不绑工具的实例）"

    if cfg.model.base_url and cfg.model.api_key:
        model = build_model(cfg.model, name=resolved_model_name, provider=args.provider)
        model_shown = describe_model(cfg)
        if not args.no_summarizer:
            # 摘要模型**必须是另一个实例，而且没绑工具**：`ContextManager.summarize` 直接调它，
            # 绑了工具就会把整份工具清单一起发出去（既贵又跑偏）。所以用配置再造一个。
            summarizer = build_model(cfg.model, name=summarizer_name, provider=args.provider)
    else:
        # 只可能是 `--show-capabilities` 那条路（正常路径在 `load_config` 就报错了）。
        model_shown = "(没配 base_url / api_key)"

    # 到这一步才真的要动这个工作区了 —— 不存在就先说一声（下面 `Harness(...)` 会把它建出来）。
    announce_new_workspace(workspace)
    harness = make_harness(args, cfg, workspace, model, summarizer=summarizer,
                           subagents=subagents, mcp_servers=mcp_servers)

    if args.show_capabilities:
        print_capabilities(harness, cfg, model_shown=model_shown,
                           summarizer_shown=summarizer_shown, mcp_servers=mcp_servers,
                           note=config_note)
        return 0

    try:
        session = harness.open(args.session)
    except Exception as exc:                        # noqa: BLE001  会话不存在 / 属于别的工作区
        print(f"打不开会话：{exc}", file=sys.stderr)
        return 2

    print(f"配置：{cfg.path}")
    print(f"模型：{model_shown}")
    try:
        try:
            from tui.app import HarnessTui
        except ImportError as exc:                  # 界面框架是可选依赖
            print(f"缺少界面依赖：{exc}\n装一下：pip install textual", file=sys.stderr)
            return 2
        HarnessTui(harness, session,
                   fold_chars=args.fold_chars or cfg.fold_chars).run()
    finally:
        harness.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
