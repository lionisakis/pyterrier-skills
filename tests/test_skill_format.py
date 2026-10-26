"""Every skill follows the Agent Skills format and this repo's size rules."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from skill_files import (
    REPO_ROOT,
    find_skills,
    read_frontmatter,
    skill_name,
    stray_skill_files,
)

NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
MAX_NAME = 64
MAX_DESCRIPTION = 1024
MAX_LINES = 500
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
# "the `pyterrier-x` skill" and "see pyterrier-x" name another skill; it must exist. Bare names in
# parentheses are usually pip packages (pyterrier-dr, pyterrier-pisa), so they are not checked.
SKILL_REF_RE = re.compile(r"`?(pyterrier-[a-z0-9-]+)`? skill\b|\bsee (pyterrier-[a-z0-9-]+)\b")

SKILLS = find_skills()
skill_param = pytest.mark.parametrize("skill", SKILLS, ids=[skill_name(s) for s in SKILLS])


@skill_param
def test_name_matches_folder(skill: Path):
    name = read_frontmatter(skill).get("name")
    assert name == skill_name(skill), f"name {name!r} must equal folder {skill_name(skill)!r}"
    assert NAME_RE.fullmatch(name), f"name {name!r}: lowercase letters, digits, single hyphens"
    assert len(name) <= MAX_NAME, f"name is {len(name)} chars, max {MAX_NAME}"


@skill_param
def test_description(skill: Path):
    description = read_frontmatter(skill).get("description")
    assert isinstance(description, str) and description.strip(), "description is required"
    assert len(description) <= MAX_DESCRIPTION, (
        f"description is {len(description)} chars, max {MAX_DESCRIPTION}"
    )


@skill_param
def test_compatibility(skill: Path):
    compatibility = read_frontmatter(skill).get("compatibility")
    assert isinstance(compatibility, str) and compatibility.strip(), (
        "compatibility is required, e.g. 'pyterrier>=1.1'"
    )


@skill_param
def test_length(skill: Path):
    lines = len(skill.read_text(encoding="utf-8").splitlines())
    assert lines < MAX_LINES, f"SKILL.md has {lines} lines; move detail into references/"


@skill_param
def test_relative_links_exist(skill: Path):
    missing = []
    for target in LINK_RE.findall(skill.read_text(encoding="utf-8")):
        if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
            continue  # URL or in-page anchor
        path = (skill.parent / target.split("#", 1)[0]).resolve()
        if not path.exists():
            missing.append(target)
    assert not missing, f"links to missing files: {missing}"


@skill_param
def test_skill_references_exist(skill: Path):
    """Every skill named in a SKILL.md or references/*.md file is a folder under skills/ or ext-skills/."""
    known = {skill_name(s) for s in SKILLS}
    missing = set()
    for path in [skill, *sorted(skill.parent.glob("references/*.md"))]:
        for match in SKILL_REF_RE.finditer(path.read_text(encoding="utf-8")):
            name = next(g for g in match.groups() if g)
            if name not in known:
                missing.add(f"{path.relative_to(REPO_ROOT)}: {name}")
    assert not missing, f"references to skills that don't exist: {sorted(missing)}"


def test_no_stray_skill_files():
    stray = [str(p.relative_to(REPO_ROOT)) for p in stray_skill_files()]
    assert not stray, (
        f"SKILL.md outside skills/<name>/ would shadow or duplicate skills: {stray}"
    )
