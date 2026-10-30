"""pt_docs/ (this repo's page in the PyTerrier docs) is complete and its generated skills table is current."""

from __future__ import annotations

import importlib.util
import re
import sys

from skill_files import REPO_ROOT

PT_DOCS = REPO_ROOT / "pt_docs"

spec = importlib.util.spec_from_file_location("build_pt_docs", REPO_ROOT / "scripts" / "build_pt_docs.py")
build_pt_docs = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = build_pt_docs
spec.loader.exec_module(build_pt_docs)


def test_skills_table_is_current():
    assert (PT_DOCS / "skills.rst").read_text(encoding="utf-8") == build_pt_docs.render(), (
        "pt_docs/skills.rst is out of date: run python scripts/build_pt_docs.py"
    )


def test_every_page_is_in_the_toctree():
    index = (PT_DOCS / "index.rst").read_text(encoding="utf-8")
    entries = set(re.findall(r"^\s+(?:[^<\n]*<)?([\w-]+)>?\s*$", index.split(".. toctree::", 1)[1], re.M))
    pages = {p.stem for p in PT_DOCS.glob("*.rst")} - {"index"}
    assert pages <= entries, f"pages missing from the toctree in pt_docs/index.rst: {sorted(pages - entries)}"


def test_rst_text_escapes_markup():
    assert build_pt_docs.rst_text("operators (>>, **, |) and `pt.apply`") == r"operators (>>, \*\*, \|) and ``pt.apply``"
