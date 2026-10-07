"""Code in skills is checked in two tiers.

* Every ```python block must parse. Fragments are allowed.
* Blocks fenced ```python run are executed in file order, in one namespace per skill
  (`pytest -m run`). Blocks fenced ```python run slow are added only under `pytest -m slow`.
  A fast `run` block must not depend on a `slow` one.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from skill_files import CodeBlock, find_skills, python_blocks, skill_name

SKILLS = find_skills()

PARSE_CASES = [(s, b) for s in SKILLS for b in python_blocks(s)]


@pytest.mark.parametrize(
    ("skill", "block"),
    PARSE_CASES,
    ids=[f"{skill_name(s)}:L{b.line}" for s, b in PARSE_CASES],
)
def test_python_block_parses(skill: Path, block: CodeBlock):
    try:
        ast.parse(block.source, filename=f"{skill}:L{block.line}")
    except SyntaxError as err:
        pytest.fail(f"{skill_name(skill)} line {block.line + (err.lineno or 0)}: {err.msg}")


def _run_blocks(skill: Path, include_slow: bool) -> list[CodeBlock]:
    return [
        b
        for b in python_blocks(skill)
        if "run" in b.flags and (include_slow or "slow" not in b.flags)
    ]


RUN_CASES = []
for s in SKILLS:
    if _run_blocks(s, include_slow=False):
        RUN_CASES.append(pytest.param(s, False, marks=pytest.mark.run, id=skill_name(s)))
    if any("slow" in b.flags for b in _run_blocks(s, include_slow=True)):
        RUN_CASES.append(
            pytest.param(s, True, marks=pytest.mark.slow, id=f"{skill_name(s)}-slow")
        )


@pytest.mark.parametrize(("skill", "include_slow"), RUN_CASES)
def test_run_blocks_execute(skill: Path, include_slow: bool, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # indices and result files land in a temp folder
    namespace: dict = {"__name__": "__main__"}
    for block in _run_blocks(skill, include_slow):
        code = compile(block.source, f"{skill_name(skill)}/SKILL.md:L{block.line}", "exec")
        exec(code, namespace)
