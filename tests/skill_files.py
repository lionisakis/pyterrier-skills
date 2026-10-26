"""Helpers shared by the skill tests: locate skills, parse frontmatter, extract code blocks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO_ROOT / "skills"
# Skills copied from extension packages' pt_skills/ by scripts/collect_ext_skills.py. They live under
# skills/ because the `skills` CLI only scans skills/ (three levels deep), the repo root and agent folders.
EXT_SKILLS_DIR = SKILLS_DIR / "ext"

# Folders never scanned for stray SKILL.md files.
IGNORED_DIRS = {".git", ".venv", "node_modules", ".pytest_cache", "__pycache__"}

_FENCE = re.compile(r"^(?P<indent> {0,3})(?P<fence>`{3,}|~{3,})(?P<info>.*)$")


@dataclass
class CodeBlock:
    """A fenced code block. `flags` holds the info-string words after the language."""

    lang: str
    flags: set[str] = field(default_factory=set)
    source: str = ""
    line: int = 0  # 1-based line of the opening fence


def find_skills() -> list[Path]:
    """Every skills/<name>/SKILL.md and collected skills/ext/<name>/SKILL.md, sorted by skill name.

    Names never repeat: the collector deletes skills/<name>/ when an extension ships the same name
    (test_no_duplicate_skill_names), because the CLI installs whichever duplicate it reads first.
    """
    return sorted([*SKILLS_DIR.glob("*/SKILL.md"), *EXT_SKILLS_DIR.glob("*/SKILL.md")],
                  key=lambda p: (p.parent.name, p))


def skill_name(path: Path) -> str:
    return path.parent.name


def read_frontmatter(path: Path) -> dict:
    """Parse the YAML frontmatter between the opening `---` lines. Raises ValueError if absent."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{path}: file must start with a '---' frontmatter line")
    for end, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            data = yaml.safe_load("\n".join(lines[1:end])) or {}
            if not isinstance(data, dict):
                raise ValueError(f"{path}: frontmatter must be a YAML mapping")
            return data
    raise ValueError(f"{path}: frontmatter has no closing '---' line")


def code_blocks(path: Path) -> list[CodeBlock]:
    """All fenced code blocks in a Markdown file, in order."""
    blocks: list[CodeBlock] = []
    current: CodeBlock | None = None
    fence = ""
    body: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if current is None:
            m = _FENCE.match(line)
            if m:
                fence = m["fence"]
                words = m["info"].split()
                current = CodeBlock(
                    lang=words[0].lower() if words else "",
                    flags={w.lower() for w in words[1:]},
                    line=lineno,
                )
                body = []
        else:
            stripped = line.strip()
            if stripped.startswith(fence[0] * len(fence)) and set(stripped) == {fence[0]}:
                current.source = "\n".join(body) + "\n"
                blocks.append(current)
                current = None
            else:
                body.append(line)
    return blocks


def python_blocks(path: Path) -> list[CodeBlock]:
    return [b for b in code_blocks(path) if b.lang in {"python", "py"}]


def stray_skill_files() -> list[Path]:
    """SKILL.md files anywhere except skills/<name>/SKILL.md and skills/ext/<name>/SKILL.md."""
    allowed = set(find_skills())
    found = []
    for path in REPO_ROOT.rglob("SKILL.md"):
        if IGNORED_DIRS.intersection(path.relative_to(REPO_ROOT).parts):
            continue
        if path not in allowed:
            found.append(path)
    return sorted(found)
