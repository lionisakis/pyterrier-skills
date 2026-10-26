"""scripts/collect_ext_skills.py copies extensions' pt_skills/ folders into ext-skills/ correctly."""

from __future__ import annotations

import importlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from skill_files import REPO_ROOT

spec = importlib.util.spec_from_file_location("collect_ext_skills", REPO_ROOT / "scripts" / "collect_ext_skills.py")
collector = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = collector  # dataclasses look the module up while it executes
spec.loader.exec_module(collector)


def skill_md(name: str, body: str = "Body.\n") -> str:
    return f"---\nname: {name}\ndescription: Use when testing.\ncompatibility: pyterrier>=1.1\n---\n\n{body}"


def make_package(root: Path, package: str, skills: dict[str, str]) -> None:
    """A minimal installed package: <package>/pt_skills/<name>/SKILL.md plus a .dist-info."""
    (root / package).mkdir(parents=True)
    (root / package / "__init__.py").write_text("")
    for name, text in skills.items():
        (root / package / "pt_skills" / name / "references").mkdir(parents=True)
        (root / package / "pt_skills" / name / "SKILL.md").write_text(text)
        (root / package / "pt_skills" / name / "references" / "api.md").write_text("# API\n")
    dist = root / f"{package}-0.1.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {package}\nVersion: 0.1\n")


@pytest.fixture
def site(tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.syspath_prepend(str(site))
    yield site
    for module in [m for m in sys.modules if m.startswith("fakeext")]:
        del sys.modules[module]
    importlib.invalidate_caches()


def run(site: Path, out: Path, lines: str) -> dict:
    importlib.invalidate_caches()
    found = []
    for ext in collector.parse_extensions(lines):
        found += collector.installed_skills(ext)[0]
    return collector.collect(found, out)


def test_parse_extensions_matches_generate_extensions():
    exts = collector.parse_extensions(
        "pyterrier_pisa # PISA <https://github.com/terrierteam/pyterrier_pisa>\n"
        "# ColBERT <https://github.com/terrierteam/pyterrier_colbert>\n\n"
        "bare_pkg\n"
    )
    assert [(e.package, e.name, e.url) for e in exts] == [
        ("pyterrier_pisa", "PISA", "https://github.com/terrierteam/pyterrier_pisa"),
        ("", "ColBERT", "https://github.com/terrierteam/pyterrier_colbert"),
        ("bare_pkg", "bare_pkg", None),
    ]


def test_installed_collects_overrides_and_skips(site, tmp_path):
    make_package(site, "fakeext", {"fake-ext": skill_md("fake-ext"), "pyterrier-basics": skill_md("pyterrier-basics")})
    out = tmp_path / "ext-skills"
    changes = run(site, out, "fakeext # Fake <https://example.org>\nnot_installed_pkg # Missing\n")

    assert (out / "fake-ext" / "SKILL.md").read_text() == skill_md("fake-ext")
    assert (out / "fake-ext" / "references" / "api.md").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert set(manifest) == {"fake-ext", "pyterrier-basics"}
    assert manifest["fake-ext"] == {**manifest["fake-ext"], "package": "fakeext", "source": "fakeext 0.1", "overrides": False}
    assert manifest["pyterrier-basics"]["overrides"] is True
    assert changes == {"added": ["fake-ext", "pyterrier-basics"], "changed": [], "removed": [], "overrides": ["pyterrier-basics"]}
    assert collector.installed_skills(collector.Extension("not_installed_pkg", "Missing", None)) == ([], "not installed")


def test_rerun_reports_changes_and_removes(site, tmp_path):
    make_package(site, "fakeext", {"fake-ext": skill_md("fake-ext"), "gone": skill_md("gone")})
    out = tmp_path / "ext-skills"
    run(site, out, "fakeext # Fake\n")

    (site / "fakeext" / "pt_skills" / "fake-ext" / "SKILL.md").write_text(skill_md("fake-ext", "New body.\n"))
    shutil.rmtree(site / "fakeext" / "pt_skills" / "gone")
    changes = run(site, out, "fakeext # Fake\n")

    assert changes["changed"] == ["fake-ext"] and changes["removed"] == ["gone"] and changes["added"] == []
    assert not (out / "gone").exists()
    assert run(site, out, "fakeext # Fake\n")["changed"] == []


def test_bad_skills_are_rejected_without_writing(site, tmp_path):
    make_package(site, "fakeext", {"fake-ext": skill_md("wrong-name")})
    out = tmp_path / "ext-skills"
    with pytest.raises(SystemExit, match="must equal the folder name"):
        run(site, out, "fakeext # Fake\n")
    assert not out.exists()

    dup = collector.Skill("same", "a", "a 1", {"SKILL.md": skill_md("same").encode()})
    with pytest.raises(SystemExit, match="shipped by both a and b"):
        collector.collect([dup, collector.Skill("same", "b", "b 1", dup.files)], out)


def test_skill_folders_finds_any_layout():
    tree = [
        {"path": "src/pkg/pt_skills/pkg-a", "type": "tree", "sha": "aaa"},
        {"path": "src/pkg/pt_skills/pkg-a/SKILL.md", "type": "blob", "sha": "1"},
        {"path": "src/pkg/pt_skills/pkg-a/references/api.md", "type": "blob", "sha": "2"},
        {"path": "pkg2/pt_skills/pkg-b", "type": "tree", "sha": "bbb"},
        {"path": "pkg2/pt_skills/pkg-b/SKILL.md", "type": "blob", "sha": "3"},
        {"path": "pkg/pt_docs/index.rst", "type": "blob", "sha": "4"},
        {"path": "skills/other/SKILL.md", "type": "blob", "sha": "5"},
    ]
    assert collector.skill_folders(tree) == {"src/pkg/pt_skills/pkg-a": "aaa", "pkg2/pt_skills/pkg-b": "bbb"}


def test_unsafe_paths_rejected():
    for path in ["../x", "/etc/passwd", "a/../../b"]:
        with pytest.raises(ValueError):
            collector.safe_relpath(path)


def test_verify_install(tmp_path):
    installed = tmp_path / "skills"
    assert any("not installed" in p for p in collector.verify_install(installed))
    for skill in collector.find_skills():
        shutil.copytree(skill.parent, installed / skill.parent.name, ignore=shutil.ignore_patterns("__pycache__"))
    assert collector.verify_install(installed) == []
    (installed / "pyterrier-basics" / "SKILL.md").write_text("changed")
    assert collector.verify_install(installed) == ["pyterrier-basics/SKILL.md: differs from skills/pyterrier-basics/SKILL.md"]


@pytest.mark.parametrize("code, result", [(404, "repo not found"), (500, SystemExit), (403, SystemExit)])
def test_github_errors_never_drop_skills_silently(monkeypatch, code, result):
    """Only a missing repo counts as "no skills"; rate limits and outages fail the run instead."""
    import urllib.error

    def fail(url, api=False):
        raise urllib.error.HTTPError(url, code, "error", {}, None)

    monkeypatch.setattr(collector, "fetch", fail)
    ext = collector.Extension("pkg", "Pkg", "https://github.com/owner/repo")
    if result is SystemExit:
        with pytest.raises(SystemExit, match=f"GitHub API error {code}"):
            collector.github_skills(ext)
    else:
        assert collector.github_skills(ext) == ([], result)
