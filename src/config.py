# -*- coding: utf-8 -*-
"""配置：**一份 YAML** ＋ 代码里的默认值。

## 三层，谁压谁

    命令行参数  >  配置文件  >  代码里的默认值

长期配置写进配置文件（`config.yml`），临时改动用命令行盖一次 —— 于是"这个 agent 到底配了
什么"有一个**看得见、留得住**的地方，而不是散在每次敲的那串参数里。

## 只有两项是必填的

``model.base_url`` 与 ``model.api_key``：少了它们连请求都发不出去，所以**当场报错**，
不猜、不静默降级 —— 猜错的后果是"跑完一批数据，事后才发现打到了别处"。
其余每一项代码里都有默认值：配置文件里没写就用默认值（**前提是配置文件本身存在**）。

## 文件不存在就报错

不给路径 → 找项目本目录下的 `config.yml`；那个文件也没有 → **报错并说清去哪找**。
刻意不"没配就用一份全默认的配置"：那样报错会推迟到第一次模型调用，而且报得很难懂
（`api_key` 为空 → 401），不如在这里一句话说清。

## 为什么不再用 ``.env``

以前 url / key 来自 ``.env``、其余来自命令行，两处都得看。现在收成一份文件。
（**评估那一侧暂时仍读 ``.env``**：迁移待办记在 `docs/eval/配置迁移.md`，本模块不碰它。）
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

#: 仓库根 = 本文件往上两层（`src/config.py`）。
REPO_ROOT = Path(__file__).resolve().parents[1]

#: 默认配置文件：**项目本目录**下这一份。
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yml"

# ---------------------------------------------------------------------------
# 代码里的默认值（配置文件里缺哪项就用它）
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_PROVIDER = "unknown"
DEFAULT_PERMISSION_MODE = "ask"
DEFAULT_MAX_TURNS = 40
DEFAULT_MAX_CONTEXT_TOKENS = 60_000
DEFAULT_SANDBOX = False
DEFAULT_FOLD_CHARS = 400

#: 必填项的（块, 键, 说明）—— 报错时照着这个列表说缺什么。
REQUIRED_FIELDS = (("model", "base_url", "模型端点"), ("model", "api_key", "API key"))


class ConfigError(RuntimeError):
    """配置有问题（文件不在、必填项缺、类型不对）—— 一律**当场报错**，不静默降级。"""


# ---------------------------------------------------------------------------
# 配置的形状
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    """一个模型端点。``kwargs`` 原样直通给 OpenAI 客户端（``temperature`` / ``extra_body`` …）。"""

    base_url: str
    api_key: str
    name: str = DEFAULT_MODEL
    provider: str = DEFAULT_PROVIDER
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SubagentConfig:
    """一个可被委派的子代理。``description`` 是写给**主模型**看的（它靠它决定派给谁）。"""

    name: str
    description: str
    system_prompt: str


@dataclass(frozen=True)
class Config:
    """整份配置。

    **``workspace`` 与 ``session_root`` 刻意不在这里**：前者是每次跑都可能变的（命令行给，
    默认当前目录），后者是固定的（`<仓库根>/sessions`，需要改的话用 `--session-root`）。
    把它们放进配置文件，等于让"这次动哪个项目"变成一次容易忘记改的全局状态。
    """

    model: ModelConfig
    path: Path
    summarizer_model: str = ""              # 空 = 与主模型相同（另起一个不绑工具的实例）
    permission_mode: str = DEFAULT_PERMISSION_MODE
    max_turns: int = DEFAULT_MAX_TURNS
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    sandbox: bool = DEFAULT_SANDBOX
    skills_dir: str = ""                    # 空 = 看 <工作区>/.harness/skills
    subagents: tuple[SubagentConfig, ...] = ()
    fold_chars: int = DEFAULT_FOLD_CHARS    # 思考块折叠阈值（工具结果另有行数规则）
    #: **MCP server 配置文件的路径**。``None`` = 不用 MCP（这是默认）。
    #: 相对路径按**这份主配置所在目录**解析（配置文件之间互相引用，就近找最不容易错）。
    mcp_config: Path | None = None


# ---------------------------------------------------------------------------
# 读
# ---------------------------------------------------------------------------
def config_path_of(path: str | os.PathLike | None = None) -> Path:
    """配置文件的落点：给了就用给的，没给就用项目本目录那份。"""
    return Path(path) if path else DEFAULT_CONFIG_PATH


def load_config(path: str | os.PathLike | None = None) -> Config:
    """读配置。**文件不存在直接报错**（见模块说明：不猜、不用一份全默认的配置兜底）。"""
    target = config_path_of(path)
    if not target.is_file():
        hint = ("" if path else f"（默认找 {DEFAULT_CONFIG_PATH}；也可以用 --config 指到别处）")
        raise ConfigError(f"配置文件不存在：{target}{hint}")

    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"配置文件解析不了：{target}\n{exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"配置文件顶层应该是一组键值对，收到 {type(raw).__name__}：{target}")
    return _from_mapping(raw, target)


def default_config(path: Path | None = None) -> Config:
    """一份"全取代码默认值"的配置，**只用于展示能力面**。

    ``load_config`` 永远不会给你这个 —— 一份没有 ``base_url`` / ``api_key`` 的配置是跑不起来的。
    它只服务于"我想看看默认会装配成什么样"这类只读场合（`--show-capabilities`）。
    """
    return Config(model=ModelConfig(base_url="", api_key=""), path=path or DEFAULT_CONFIG_PATH)


def _from_mapping(raw: dict[str, Any], path: Path) -> Config:
    model_block = _block(raw, "model", path)

    base_url = _text(model_block, "base_url")
    api_key = _text(model_block, "api_key")
    missing = [f"{block}.{key}（{label}）" for block, key, label in REQUIRED_FIELDS
               if not _text(_block(raw, block, path), key)]
    if missing:
        raise ConfigError(f"{path} 里缺了必填项：{'、'.join(missing)}")

    harness_block = _block(raw, "harness", path)
    capabilities = _block(raw, "capabilities", path)
    ui = _block(raw, "ui", path)

    return Config(
        model=ModelConfig(
            base_url=base_url,
            api_key=api_key,
            name=_text(model_block, "name") or DEFAULT_MODEL,
            provider=_text(model_block, "provider") or DEFAULT_PROVIDER,
            kwargs=_block(model_block, "kwargs", path),
        ),
        path=path,
        summarizer_model=_text(_block(raw, "summarizer", path), "name"),
        permission_mode=_text(harness_block, "permission_mode") or DEFAULT_PERMISSION_MODE,
        max_turns=_count(harness_block, "max_turns", DEFAULT_MAX_TURNS, path),
        max_context_tokens=_count(harness_block, "max_context_tokens",
                                  DEFAULT_MAX_CONTEXT_TOKENS, path),
        sandbox=_flag(capabilities, "sandbox", DEFAULT_SANDBOX, path),
        skills_dir=_text(capabilities, "skills_dir"),
        subagents=_subagents(capabilities, path),
        fold_chars=_count(ui, "fold_chars", DEFAULT_FOLD_CHARS, path),
        mcp_config=_mcp_config_path(raw, path),
    )


def _mcp_config_path(raw: dict[str, Any], path: Path) -> Path | None:
    """``mcp.config`` → 绝对路径；空/没写就是 ``None``（＝不用 MCP）。

    **相对路径按主配置所在目录解析**（不是当前工作目录、也不是仓库根）：配置文件之间互相引用时，
    "就在我旁边"是最不容易错的读法 —— 项目挪个地方，这份引用跟着走。
    """
    text = _text(_block(raw, "mcp", path), "config")
    if not text:
        return None
    candidate = Path(text)
    return candidate if candidate.is_absolute() else (path.parent / candidate)


# ---------------------------------------------------------------------------
# MCP：读那份 server 配置文件，**在启动时校验并造出对象**
# ---------------------------------------------------------------------------
#: MCP 配置文件里必须有的顶层键（沿用 Claude Desktop 那套写法，行业里到处是它）。
MCP_SERVERS_KEY = "mcpServers"

#: 每个 server 允许的键。**多出来的键一律拒绝**：写错了名字（比如 `cmd`）却被静默忽略，
#: 后果是"配置看起来生效了、其实没生效"，那比报错难查得多。
MCP_SERVER_KEYS = ("command", "args", "env", "timeout_s")


def build_mcp_servers(cfg: Config) -> list[Any]:
    """按 ``cfg.mcp_config`` 读 MCP 配置文件 → `MCPServerStdio` 清单。

    **没配路径就是不用 MCP**（返回空清单，一个文件都不碰）。
    只要给了路径，就**当场把整份文件校验完**：文件在不在、能不能解析（JSON 也行 ——
    JSON 是 YAML 的子集，同一个 loader 吃两种写法）、顶层有没有 ``mcpServers``、
    每个条目有没有 ``command``、类型对不对、命令找不找得到。

    任何一条不过就抛 :class:`ConfigError`（调用方负责"打警告并停止启动"）。
    **刻意不在启动时连 server**：那是副作用（要起子进程、要握手），由 `Harness` 打开会话时做；
    这里只保证"这份配置本身是对的"，让错误在**几毫秒内**暴露，而不是等模型跑到一半才发现。
    """
    if cfg.mcp_config is None:
        return []

    path = cfg.mcp_config
    if not path.is_file():
        raise ConfigError(f"MCP 配置文件不存在：{path}\n"
                          f"（配置里 `mcp.config` 指向了它；不想用 MCP 就把那一行留空）")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"MCP 配置文件解析不了：{path}\n{exc}") from exc
    except OSError as exc:
        raise ConfigError(f"MCP 配置文件读不动：{path}\n{exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"MCP 配置文件顶层应该是一组键值对（要有 {MCP_SERVERS_KEY}），"
                          f"收到 {type(raw).__name__}：{path}")
    servers = raw.get(MCP_SERVERS_KEY)
    if servers is None:
        raise ConfigError(f"MCP 配置里没有 {MCP_SERVERS_KEY}：{path}\n"
                          f"（写法：{MCP_SERVERS_KEY}: 下面一行一个 server）")
    if not isinstance(servers, dict):
        raise ConfigError(f"{path} 的 {MCP_SERVERS_KEY} 应该是一组键值对"
                          f"（键 = server 名），收到 {type(servers).__name__}")
    if not servers:
        raise ConfigError(f"{path} 里 {MCP_SERVERS_KEY} 是空的 —— 给了配置文件却一个 server 都没有，"
                          f"多半是写漏了（不想用 MCP 就把 `mcp.config` 留空）")

    built = [_mcp_server(name, spec, path) for name, spec in servers.items()]
    return built


def _mcp_server(name: Any, spec: Any, path: Path) -> Any:
    """一个 server 条目 → `MCPServerStdio`（顺带把该查的都查了）。"""
    from src.harness.mcp import MCPServerStdio          # 延迟导入：读配置不必拉起这一层

    label = f"{path} 的 {MCP_SERVERS_KEY}.{name}"
    if not str(name).strip():
        raise ConfigError(f"{path} 的 {MCP_SERVERS_KEY} 里有一个空名字的 server")
    if not isinstance(spec, dict):
        raise ConfigError(f"{label}: 应该是一组键值对（command / args / env / timeout_s），"
                          f"收到 {type(spec).__name__}")

    unknown = [key for key in spec if key not in MCP_SERVER_KEYS]
    if unknown:
        raise ConfigError(f"{label}: 不认识的键 {unknown}（只认 {list(MCP_SERVER_KEYS)}）"
                          f"—— 写错名字被静默忽略，比报错难查得多")

    command = spec.get("command")
    args = spec.get("args")
    if isinstance(command, list):
        if args is not None:
            raise ConfigError(f"{label}: `command` 已经是整条命令（列表），就不要再给 `args` 了"
                              f"—— 两处都写，谁接谁不清楚")
        argv = [_one_text(item, f"{label} 的 command") for item in command]
    elif isinstance(command, str) and command.strip():
        argv = [command.strip()]
        if args is None:
            args = []
        if not isinstance(args, list):
            raise ConfigError(f"{label}: `args` 应该是列表，收到 {type(args).__name__}")
        argv += [_one_text(item, f"{label} 的 args") for item in args]
    elif command is None:
        raise ConfigError(f"{label}: 缺 `command`（要启动的命令，比如 `npx` 或某个可执行文件路径）")
    else:
        raise ConfigError(f"{label}: `command` 应该是字符串（再配 `args`）或整个列表，"
                          f"收到 {type(command).__name__}")
    if not argv or not argv[0]:
        raise ConfigError(f"{label}: 命令是空的")

    _check_command(argv[0], label)
    env = _mcp_env(spec.get("env"), label)
    timeout_s = _positive(spec.get("timeout_s"), label, default=30.0)

    return MCPServerStdio(command=argv, name=str(name), timeout_s=timeout_s, env=env)


def _one_text(value: Any, label: str) -> str:
    if isinstance(value, (dict, list)):
        raise ConfigError(f"{label}: 每一项都应该是字符串，收到 {type(value).__name__}")
    return str(value)


def _mcp_env(raw: Any, label: str) -> dict[str, str] | None:
    """``env`` → 字符串字典（不给就 ``None``：**完全继承父进程环境**）。"""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError(f"{label}: `env` 应该是一组键值对，收到 {type(raw).__name__}")
    out: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(value, (dict, list)):
            raise ConfigError(f"{label} 的 env.{key}: 应该是字符串，收到 {type(value).__name__}")
        out[str(key)] = str(value)
    return out


def _positive(raw: Any, label: str, *, default: float) -> float:
    if raw is None:
        return default
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ConfigError(f"{label}: `timeout_s` 应该是数字，收到 {raw!r}")
    if raw <= 0:
        raise ConfigError(f"{label}: `timeout_s` 应该是正数，收到 {raw!r}")
    return float(raw)


def _check_command(program: str, label: str) -> None:
    """命令找不找得到 —— **启动时就查**（打错一个字母，不该等到模型跑到一半才发现）。"""
    import shutil

    if os.path.isabs(program):
        if not os.path.isfile(program):
            raise ConfigError(f"{label}: 命令不存在：{program}")
        return
    if shutil.which(program) is None:
        raise ConfigError(f"{label}: 在 PATH 里找不到命令 {program!r}"
                          f"（要绝对路径也行；装好了再启动）")


# ---------------------------------------------------------------------------
# 类型检查：配置文件里的东西是**人写的**，写错了要在这里说清楚
# ---------------------------------------------------------------------------
def _block(raw: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = raw.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{path} 的 {key}: 应该是一组键值对，收到 {type(value).__name__}")
    return value


def _text(block: dict[str, Any], key: str, default: str = "") -> str:
    value = block.get(key)
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        raise ConfigError(f"{key}: 应该是一个值，收到 {type(value).__name__}")
    return str(value).strip()


def _count(block: dict[str, Any], key: str, default: int, path: Path) -> int:
    value = block.get(key)
    if value is None:
        return default
    # bool 是 int 的子类，但 `max_turns: true` 显然不是想写 1
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{path} 的 {key}: 应该是整数，收到 {value!r}")
    if value <= 0:
        raise ConfigError(f"{path} 的 {key}: 应该是正数，收到 {value!r}")
    return value


def _flag(block: dict[str, Any], key: str, default: bool, path: Path) -> bool:
    value = block.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ConfigError(f"{path} 的 {key}: 应该是 true / false，收到 {value!r}")
    return value


def _subagents(capabilities: dict[str, Any], path: Path) -> tuple[SubagentConfig, ...]:
    items = capabilities.get("subagents")
    if items is None:
        return ()
    if not isinstance(items, list):
        raise ConfigError(f"{path} 的 capabilities.subagents: 应该是一个列表，"
                          f"收到 {type(items).__name__}")
    out: list[SubagentConfig] = []
    seen: set[str] = set()
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ConfigError(f"{path} 的 capabilities.subagents 第 {index} 项应该是一组键值对"
                              f"（name / description / system_prompt）")
        fields = {key: _text(item, key) for key in ("name", "description", "system_prompt")}
        missing = [key for key, value in fields.items() if not value]
        if missing:
            raise ConfigError(f"{path} 的 capabilities.subagents 第 {index} 项缺了："
                              f"{'、'.join(missing)}")
        if fields["name"] in seen:
            # 重名在 `make_task_tool` 里是**静默覆盖**（它按名字建 dict），所以在入口拦下
            raise ConfigError(f"{path} 的子代理名字重复：{fields['name']!r}（后者会盖掉前者）")
        seen.add(fields["name"])
        out.append(SubagentConfig(**fields))
    return tuple(out)


# ---------------------------------------------------------------------------
# 用：造模型、写一行说明
# ---------------------------------------------------------------------------
def build_model(cfg: ModelConfig, *, name: str | None = None, provider: str | None = None,
                override: dict[str, Any] | None = None) -> Any:
    """配置 → 一个真的 ``OpenAIChatModel``。

    **每次调用都造一个新的**：摘要模型必须是"另一个没绑工具的实例"（`ContextManager.summarize`
    直接调它，绑了工具就会把整份工具清单一起发出去），所以这里刻意不做缓存。
    """
    from src.core.models import OpenAIChatModel      # 延迟导入：读配置不必装 openai

    return OpenAIChatModel(model=name or cfg.name,
                           base_url=cfg.base_url,
                           api_key=cfg.api_key,
                           provider=provider or cfg.provider,
                           **{**cfg.kwargs, **(override or {})})


def describe_model(cfg: Config) -> str:
    """给人和日志看的一行：**provider / model / base_url，没有 key**（key 只进不出）。"""
    return (f"provider={cfg.model.provider} model={cfg.model.name} "
            f"base_url={cfg.model.base_url} API_KEY=已配置")
