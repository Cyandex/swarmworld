#!/usr/bin/env python3
"""Verify that a SwarmWorld source release is clean, linked, and reproducible."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "RELEASE_MANIFEST.sha256"

ALLOWED_TOP_LEVEL = {
    ".agents",
    ".git",
    ".gitignore",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "DATA.md",
    "LICENSE",
    "Makefile",
    "README.md",
    "RELEASE_MANIFEST.sha256",
    "RELEASE_PROVENANCE.json",
    "SECURITY.md",
    "configs",
    "docs",
    "environment.yml",
    "game",
    "godot",
    "pyproject.toml",
    "runs",
    "scripts",
    "src",
    "tests",
    "web",
    "worlds",
}

ALLOWED_AGENT_ENTRIES = {
    ".agents/skills",
    ".agents/skills/swarmworld-world-builder",
    ".agents/skills/swarmworld-world-builder/SKILL.md",
    ".agents/skills/swarmworld-world-builder/agents",
    ".agents/skills/swarmworld-world-builder/agents/openai.yaml",
    ".agents/skills/swarmworld-world-builder/references",
    ".agents/skills/swarmworld-world-builder/references/format-v1-contract.md",
    ".agents/skills/swarmworld-world-builder/references/verification-workflow.md",
}

REQUIRED = {
    "README.md",
    "LICENSE",
    "DATA.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "RELEASE_PROVENANCE.json",
    "pyproject.toml",
    "environment.yml",
    "src/biofoundry",
    "configs/demo.yaml",
    "tests",
    "game/tests",
    "web/package.json",
    "web/package-lock.json",
    "worlds/ashen_realms",
    "runs/.gitkeep",
    ".agents/skills/swarmworld-world-builder/SKILL.md",
}

FORBIDDEN_TOP_LEVEL = {
    ".claude",
    "data_share",
    "journal_figures",
    "media",
    "movies",
    "movies-final",
    "movies-new",
    "output",
    "reports",
    "tmp",
}

FORBIDDEN_DIR_NAMES = {
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
}

FORBIDDEN_FILE_SUFFIXES = {
    ".aux",
    ".bbl",
    ".bcf",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".pyc",
    ".pyo",
    ".run.xml",
    ".synctex.gz",
}

SECRET_PATTERNS = {
    "OpenAI-style secret": re.compile(r"s" + r"k-[A-Za-z0-9_-]{20,}"),
    "GitHub token": re.compile(r"gh" + r"[pousr]_[A-Za-z0-9]{20,}"),
    "AWS access key": re.compile(r"AK" + r"IA[0-9A-Z]{16}"),
    "Slack token": re.compile(r"xo" + r"[abprs]-[A-Za-z0-9-]{10,}"),
    "private key": re.compile(r"BEGIN [A-Z ]*PRIVATE" + r" KEY"),
}

ABSOLUTE_PATH_PATTERNS = {
    "macOS user path": re.compile("/" + "Users/"),
    "Linux home path": re.compile("/" + "home/"),
    "Windows user path": re.compile(r"[A-Za-z]:\\" + r"Users\\"),
}

MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def iter_release_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(ROOT).parts
    )


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_content(path: Path) -> str | None:
    if path.stat().st_size > 5 * 1024 * 1024:
        return None
    data = path.read_bytes()
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def check_structure(errors: list[str]) -> None:
    present = {path.name for path in ROOT.iterdir()}
    for name in sorted(present - ALLOWED_TOP_LEVEL):
        errors.append(f"unexpected top-level entry: {name}")
    for name in sorted(FORBIDDEN_TOP_LEVEL & present):
        errors.append(f"forbidden top-level entry: {name}")
    for name in sorted(REQUIRED):
        if not (ROOT / name).exists():
            errors.append(f"missing required path: {name}")

    agents_root = ROOT / ".agents"
    if agents_root.exists():
        agent_entries = {relative(path) for path in agents_root.rglob("*")}
        for name in sorted(agent_entries - ALLOWED_AGENT_ENTRIES):
            errors.append(f"unexpected agent metadata: {name}")
        for name in sorted(ALLOWED_AGENT_ENTRIES - agent_entries):
            errors.append(f"missing approved agent-skill entry: {name}")

    for path in ROOT.rglob("*"):
        if ".git" in path.relative_to(ROOT).parts:
            continue
        if path.is_symlink():
            errors.append(f"symbolic link not allowed in release: {relative(path)}")
        if path.is_dir() and (path.name in FORBIDDEN_DIR_NAMES or path.name.endswith(".egg-info")):
            errors.append(f"forbidden generated directory: {relative(path)}")
        if path.is_file():
            if path.name in {".DS_Store", ".env", ".env.local"}:
                errors.append(f"forbidden local file: {relative(path)}")
            if any(path.name.endswith(suffix) for suffix in FORBIDDEN_FILE_SUFFIXES):
                errors.append(f"forbidden generated file: {relative(path)}")


def check_text(errors: list[str]) -> None:
    for path in iter_release_files():
        text = text_content(path)
        if text is None:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"possible {label} in {relative(path)}")
        for label, pattern in ABSOLUTE_PATH_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"machine-specific {label} in {relative(path)}")


def normalized_link_target(raw: str) -> str:
    target = raw.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    elif " " in target:
        target = target.split(" ", 1)[0]
    return unquote(target).split("#", 1)[0]


def check_markdown_links(errors: list[str]) -> None:
    for path in iter_release_files():
        if path.suffix.lower() not in {".md", ".markdown"}:
            continue
        text = text_content(path) or ""
        for raw in MARKDOWN_LINK.findall(text):
            target = normalized_link_target(raw)
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            candidate = (path.parent / target).resolve()
            try:
                candidate.relative_to(ROOT)
            except ValueError:
                errors.append(f"link escapes release in {relative(path)}: {raw}")
                continue
            if not candidate.exists():
                errors.append(f"broken local link in {relative(path)}: {raw}")


def write_manifest() -> None:
    lines = [
        f"{sha256(path)}  {relative(path)}"
        for path in iter_release_files()
        if path != MANIFEST
    ]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_manifest(errors: list[str]) -> None:
    if not MANIFEST.exists():
        return
    expected: dict[str, str] = {}
    for line_number, line in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            digest, name = line.split("  ", 1)
        except ValueError:
            errors.append(f"invalid manifest line {line_number}")
            continue
        expected[name] = digest

    actual = {
        relative(path): sha256(path)
        for path in iter_release_files()
        if path != MANIFEST
    }
    for name in sorted(set(expected) - set(actual)):
        errors.append(f"manifest file missing: {name}")
    for name in sorted(set(actual) - set(expected)):
        errors.append(f"file absent from manifest: {name}")
    for name in sorted(set(expected) & set(actual)):
        if expected[name] != actual[name]:
            errors.append(f"manifest hash mismatch: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="write RELEASE_MANIFEST.sha256 before verification",
    )
    args = parser.parse_args()

    if args.write_manifest:
        write_manifest()

    errors: list[str] = []
    check_structure(errors)
    check_text(errors)
    check_markdown_links(errors)
    check_manifest(errors)

    if errors:
        print(f"Release verification failed with {len(errors)} issue(s):", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"Release verification passed: {len(iter_release_files())} files checked.")
    if not MANIFEST.exists():
        print("No manifest found; run with --write-manifest after validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
