import os
import re
from dataclasses import dataclass

from src.core.tool import Tool

SKILL_FILE = "SKILL.md"

#: ``read_skill`` 的结果上限（字符）。手册按需加载，别按工具默认的 2000 拦腰截断（L4）。
DEFAULT_SKILL_MAX_CHARS = 32_000

#: 单个 ``SKILL.md`` 正文的上限（字节）。超过就截断并留说明。
MAX_SKILL_BODY_BYTES = 64 * 1024

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)

@dataclass
class Skill:
    """一份技能。``body`` 是剥掉 frontmatter 之后的正文（加载时读一次，不再重读磁盘）。"""

    name: str
    description: str
    path: str
    body: str = ""

def _parse_skill(path: str) -> Skill | None:
    """从一份 ``SKILL.md`` 解析出技能：frontmatter 的 ``name`` / ``description`` 优先，
    没写就退回"目录名 + 正文第一段"。**读不了（非 UTF-8 / IO 错误）返回 ``None``** ——
    坏文件直接跳过，不把乱码塞进模型的上下文。"""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except (UnicodeDecodeError, OSError):
        return None

    name = os.path.basename(os.path.dirname(path))
    description = ""
    match = _FRONTMATTER.match(text)
    if match:
        for line in match.group(1).splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "name" and value.strip():
                name = value.strip()
            elif key.strip() == "description":
                description = value.strip()
        body = _FRONTMATTER.sub("", text).strip()
    else:
        body = text.strip()

    if not description:                    # 没写描述就拿正文第一段凑合
        description = body.splitlines()[0][:120] if body else "(无描述)"
    if len(body) > MAX_SKILL_BODY_BYTES:
        body = body[:MAX_SKILL_BODY_BYTES] + "\n…[技能正文超过上限，已截断]"
    return Skill(name=name, description=description, path=path, body=body)

def load_skills(skills_dir: str) -> list[Skill]:
    """扫描 ``skills_dir/*/SKILL.md``，返回技能清单。目录不存在返回空；坏文件跳过。"""
    out: list[Skill] = []
    if not os.path.isdir(skills_dir):
        return out
    for entry in sorted(os.listdir(skills_dir)):
        candidate = os.path.join(skills_dir, entry, SKILL_FILE)
        if os.path.isfile(candidate):
            skill = _parse_skill(candidate)
            if skill is not None:
                out.append(skill)
    return out

def skills_section(skills: list[Skill]) -> str:
    """生成注入 system prompt 的"第一级披露"：只有名字和一句话描述。"""
    if not skills:
        return ""
    lines = "\n".join(f"- {s.name}: {s.description}" for s in skills)
    return (
        "# 可用技能\n"
        "以下技能是可按需加载的操作手册。当任务与某技能的描述匹配时，"
        "先调用 read_skill 读取其全文，再按其中的指引行事：\n" + lines)

def make_skill_tool(skills: list[Skill]) -> Tool:
    """第二级披露的入口：``read_skill(name)`` 把技能正文拉进上下文。"""
    by_name = {s.name: s for s in skills}

    def read_skill(name: str) -> str:
        skill = by_name.get(name)
        if skill is None:
            return f"[错误] 没有名为 {name!r} 的技能，可用: {', '.join(by_name) or '(无)'}"
        return f"<skill name={skill.name!r} dir={os.path.dirname(skill.path)!r}>\n" \
               f"{skill.body}\n</skill>"

    return Tool.from_schema(
        name="read_skill",
        description="加载一个技能的完整操作手册。技能清单见 system prompt 的\"可用技能\"一节。",
        parameters={"type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"]},
        func=read_skill,
        max_result_chars=DEFAULT_SKILL_MAX_CHARS,
    )
