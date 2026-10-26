#!/usr/bin/env python3
"""Collect the agent skills that PyTerrier extensions ship in a `pt_skills/` folder into `skills/ext/`.

Works like PyTerrier's `docs/extras/generate_extensions.py`, which copies each extension's `pt_docs/`
folder into the readthedocs build. It reads the same `docs/extensions.txt` (`package # Name <url>` per
line) and looks for `pt_skills/<name>/SKILL.md` in each extension:

    --from github      (default) list each extension's GitHub repo (the `<url>`) and download every
                       `**/pt_skills/<name>/` folder from its default branch. Needs no pip installs.
    --from installed   look in the pip-installed package, as the docs build does:
                       `importlib.resources.files(package) / "pt_skills"`.

Each skill is copied to `skills/ext/<name>/` (under `skills/` because that is where the `skills` CLI
looks), replacing what was there; skills no longer provided are removed. Where an extension ships a
name we also have, our `skills/<name>/` is deleted, since the CLI installs whichever duplicate it reads
first. `skills/ext/manifest.json` records where each skill came from.
The run fails, writing nothing, if a SKILL.md has bad frontmatter or two extensions ship the same name.

Examples
    python scripts/collect_ext_skills.py
    python scripts/collect_ext_skills.py --from installed --extensions ../pyterrier/docs/extensions.txt
    python scripts/collect_ext_skills.py --changes /tmp/changes.md     # summary for the PR body
    python scripts/collect_ext_skills.py --verify-install-dir /tmp/p/.claude/skills
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.resources
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))
from skill_files import EXT_SKILLS_DIR, SKILLS_DIR, find_skills, read_frontmatter  # noqa: E402

EXTENSIONS_URL = "https://raw.githubusercontent.com/terrier-org/pyterrier/master/docs/extensions.txt"
USER_AGENT = "pyterrier-skills-collector/1.0"
SKILL_FILE = re.compile(r"(?:^|/)pt_skills/([^/]+)/SKILL\.md$")
GITHUB_URL = re.compile(r"https?://github\.com/([^/\s]+)/([^/\s#?]+?)(?:\.git)?/?$")
IGNORED_PARTS = {"__pycache__", ".DS_Store"}


@dataclass
class Extension:
    package: str  # pip package; empty for link-only lines ("# Name <url>")
    name: str
    url: str | None


@dataclass
class Skill:
    name: str
    package: str
    source: str  # "<package> <version>" or "<owner>/<repo>@<tree sha>"
    files: dict[str, bytes]  # path inside the skill folder -> content


def parse_extensions(text: str) -> list[Extension]:
    """Parse extensions.txt the way generate_extensions.py does."""
    extensions = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if "#" in line:
            package, name = line.split("#", 1)
        else:
            package, name = line, line
        package, name, url = package.strip(), name.strip(), None
        if "<" in name:
            name, url = name.split("<", 1)
            name, url = name.strip(), url.strip().rstrip(">")
        extensions.append(Extension(package, name, url))
    return extensions


def read_extensions(source: str) -> list[Extension]:
    if re.match(r"^https?://", source):
        return parse_extensions(fetch(source).decode("utf-8"))
    return parse_extensions(Path(source).read_text(encoding="utf-8"))


def fetch(url: str, api: bool = False) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if api:
        req.add_header("Accept", "application/vnd.github+json")
    # Only GitHub's own hosts get the token; it is what lets private repos be read at all.
    if token and urllib.parse.urlsplit(url).hostname in {"api.github.com", "raw.githubusercontent.com"}:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def safe_relpath(path: str) -> str:
    parts = PurePosixPath(path).parts
    if not parts or path.startswith("/") or ".." in parts:
        raise ValueError(f"unsafe path in skill folder: {path!r}")
    return path


# --- --from installed -----------------------------------------------------------------------------


def _walk(folder, prefix: str = ""):
    for entry in sorted(folder.iterdir(), key=lambda e: e.name):
        if entry.name in IGNORED_PARTS:
            continue
        if entry.is_dir():
            yield from _walk(entry, f"{prefix}{entry.name}/")
        else:
            yield f"{prefix}{entry.name}", entry.read_bytes()


def installed_skills(ext: Extension) -> tuple[list[Skill], str]:
    if not ext.package:
        return [], "link only"
    try:
        version = importlib.metadata.version(ext.package)
        root = importlib.resources.files(ext.package.replace("-", "_")) / "pt_skills"
    except (importlib.metadata.PackageNotFoundError, ModuleNotFoundError):
        return [], "not installed"
    if not root.is_dir():
        return [], "no pt_skills"
    skills = [
        Skill(folder.name, ext.package, f"{ext.package} {version}", dict(_walk(folder)))
        for folder in sorted(root.iterdir(), key=lambda e: e.name)
        if folder.is_dir() and (folder / "SKILL.md").is_file()
    ]
    return skills, f"{len(skills)} skill(s)"


# --- --from github --------------------------------------------------------------------------------


def skill_folders(tree: list[dict]) -> dict[str, str]:
    """Map each `pt_skills/<name>/` folder in a recursive git tree listing to its tree sha."""
    dirs = {e["path"]: e["sha"] for e in tree if e["type"] == "tree"}
    folders = {}
    for entry in tree:
        if entry["type"] == "blob" and SKILL_FILE.search(entry["path"]):
            folder = entry["path"].rsplit("/", 1)[0]
            folders[folder] = dirs.get(folder, "")
    return folders


def github_skills(ext: Extension) -> tuple[list[Skill], str]:
    m = GITHUB_URL.match(ext.url or "")
    if not m:
        return [], "no GitHub URL"
    owner, repo = m.groups()
    try:
        # HEAD is the default branch; one API call per repo keeps unauthenticated runs under the limit.
        listing = json.loads(fetch(f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1", api=True))
    except urllib.error.HTTPError as e:
        if e.code == 404:  # repo gone, renamed or not visible to the token: it ships no skills
            return [], "repo not found"
        raise SystemExit(f"{owner}/{repo}: GitHub API error {e.code}; not collecting, so its skills are not dropped")
    if listing.get("truncated"):
        print(f"warning: tree listing of {owner}/{repo} is truncated", file=sys.stderr)
    tree = listing["tree"]
    skills = []
    try:
        for folder, sha in sorted(skill_folders(tree).items()):
            prefix = folder + "/"
            files = {
                safe_relpath(e["path"][len(prefix):]): fetch(
                    f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{urllib.parse.quote(e['path'])}"
                )
                for e in tree
                if e["type"] == "blob" and e["path"].startswith(prefix)
                and not IGNORED_PARTS.intersection(PurePosixPath(e["path"]).parts)
            }
            skills.append(Skill(folder.rsplit("/", 1)[1], ext.package or ext.name, f"{owner}/{repo}@{sha[:12]}", files))
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{owner}/{repo}: download error {e.code} for {e.url}; not collecting")
    return skills, f"{len(skills)} skill(s)" if skills else "no pt_skills"


# --- writing skills/ext/ --------------------------------------------------------------------------


def digest(files: dict[str, bytes]) -> str:
    h = hashlib.sha1()
    for path in sorted(files):
        h.update(path.encode() + b"\0" + hashlib.sha1(files[path]).digest())
    return h.hexdigest()


def collect(found: list[Skill], out: Path, skills_dir: Path = SKILLS_DIR) -> dict[str, list[str]]:
    """Replace `out` with the found skills, deleting `skills_dir/<name>/` where an extension ships the
    same name (the CLI installs whichever duplicate it reads first, so only one copy may exist).

    Returns skill names: added / changed / removed (from `out`), deleted (our copies deleted now) and
    overrides (every collected skill that replaced one of ours, now or earlier).
    """
    if out.resolve() == skills_dir.resolve():
        raise SystemExit("--out must not be the skills/ folder itself")
    by_name: dict[str, Skill] = {}
    errors = []
    for skill in found:
        if skill.name in by_name:
            errors.append(f"{skill.name}: shipped by both {by_name[skill.name].package} and {skill.package}")
        else:
            by_name[skill.name] = skill

    manifest_path = out / "manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp)
        for skill in by_name.values():
            for rel, content in skill.files.items():
                dest = staged / skill.name / safe_relpath(rel)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(content)
            try:
                name = read_frontmatter(staged / skill.name / "SKILL.md").get("name")
            except ValueError as e:
                errors.append(f"{skill.name} ({skill.package}): {e}")
                continue
            if name != skill.name:
                errors.append(f"{skill.name} ({skill.package}): frontmatter name {name!r} must equal the folder name")
        if errors:
            raise SystemExit("not collecting, fix these first:\n  " + "\n  ".join(errors))

        out.mkdir(parents=True, exist_ok=True)
        for path in out.iterdir():
            if path.is_dir() and path.name not in by_name:
                shutil.rmtree(path)
        for name in by_name:
            shutil.rmtree(out / name, ignore_errors=True)
            shutil.copytree(staged / name, out / name)

    deleted = sorted(n for n in by_name if (skills_dir / n / "SKILL.md").exists())
    for name in deleted:
        shutil.rmtree(skills_dir / name)
    manifest = {
        name: {
            "package": skill.package,
            "source": skill.source,
            "hash": digest(skill.files),
            "overrides": name in deleted or old.get(name, {}).get("overrides", False),
        }
        for name, skill in sorted(by_name.items())
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "added": sorted(set(manifest) - set(old)),
        "changed": sorted(n for n in manifest if n in old and old[n]["hash"] != manifest[n]["hash"]),
        "removed": sorted(set(old) - set(manifest)),
        "deleted": deleted,
        "overrides": sorted(n for n, entry in manifest.items() if entry["overrides"]),
        "restore": sorted(n for n in set(old) - set(manifest) if old[n].get("overrides")),
    }


def changes_markdown(changes: dict[str, list[str]], manifest: dict, statuses: list[tuple[str, str]]) -> str:
    lines = ["Skills collected from extensions' `pt_skills/` by `scripts/collect_ext_skills.py`.", ""]
    for key, title in [("added", "Added"), ("changed", "Changed"), ("removed", "Removed"),
                       ("deleted", "Deleted from `skills/`: the extension's copy replaces ours"),
                       ("restore", "No longer shipped, and it had replaced ours: restore `skills/<name>/` "
                                   "from git history if we still want it")]:
        if changes[key]:
            lines.append(f"**{title}**")
            lines += [f"- `{n}`" + (f" from {manifest[n]['source']}" if n in manifest else "") for n in changes[key]]
            lines.append("")
    lines += ["<details><summary>Every extension</summary>", "", "| extension | result |", "|---|---|"]
    lines += [f"| {name} | {status} |" for name, status in statuses]
    lines += ["", "</details>", ""]
    return "\n".join(lines)


def verify_install(installed: Path) -> list[str]:
    """Problems with a `npx skills add` install of this repo: every skill present, the winning copy."""
    problems = []
    expected = {skill.parent.name: skill.parent for skill in find_skills()}
    present = {p.name for p in installed.iterdir() if (p / "SKILL.md").exists()} if installed.is_dir() else set()
    for name in sorted(present - set(expected)):
        problems.append(f"{name}: installed but not in skills/ or skills/ext/")
    for name, source in sorted(expected.items()):
        if name not in present:
            problems.append(f"{name}: not installed")
            continue
        for path in sorted(p for p in source.rglob("*") if p.is_file()):
            rel = path.relative_to(source)
            if IGNORED_PARTS.intersection(rel.parts):
                continue
            target = installed / name / rel
            if not target.is_file() or target.read_bytes() != path.read_bytes():
                problems.append(f"{name}/{rel}: differs from {path.relative_to(REPO)}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="source", choices=["github", "installed"], default="github")
    ap.add_argument("--extensions", default=EXTENSIONS_URL, help="path or URL of extensions.txt")
    ap.add_argument("--out", type=Path, default=EXT_SKILLS_DIR)
    ap.add_argument("--changes", type=Path, help="write a Markdown summary of the changes here")
    ap.add_argument("--verify-install-dir", type=Path, metavar="DIR",
                    help="only check that DIR (e.g. a project's .claude/skills) holds exactly this repo's skills")
    args = ap.parse_args()

    if args.verify_install_dir:
        problems = verify_install(args.verify_install_dir)
        for problem in problems:
            print(problem)
        print(f"{len(find_skills())} skills expected, {len(problems)} problem(s)")
        sys.exit(1 if problems else 0)

    find = github_skills if args.source == "github" else installed_skills
    found, statuses = [], []
    for ext in read_extensions(args.extensions):
        skills, status = find(ext)
        label = ext.package or ext.name
        print(f"{label:28} {status}")
        statuses.append((label, status))
        found += skills
    changes = collect(found, args.out)
    manifest = json.loads((args.out / "manifest.json").read_text(encoding="utf-8"))
    print(f"\n{len(manifest)} skill(s) in {args.out}: " + ", ".join(f"{k} {len(v)}" for k, v in changes.items()))
    if args.changes:
        args.changes.write_text(changes_markdown(changes, manifest, statuses), encoding="utf-8")


if __name__ == "__main__":
    main()
