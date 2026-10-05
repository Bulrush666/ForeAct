"""Fail when the working tree contains common private-release artifacts."""
from __future__ import annotations

import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".codex_backups", "__pycache__", ".pytest_cache"}
LOCAL_ROOTS = {"data", "pretrained", "checkpoints", "outputs"}
TEXT_LIMIT = 2_000_000

RULES = {
    "private Unix path": re.compile(
        r"/(?:data/local-data|mnt/(?![^\s]*?/data/real_inpainting)|home/CONNECT|root)(?:/|\\)"
    ),
    "Windows user path": re.compile(r"(?i)\b[A-Z]:\\Users\\"),
    "likely API token": re.compile(r"(?:hf_|sk-)[A-Za-z0-9_-]{20,}"),
    "private key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    "literal password or secret": re.compile(
        r"(?i)(?:password|passwd|api_key|access_token|secret_key)\s*[:=]\s*[\"'][^\"'\n]{4,}[\"']"
    ),
}


def iter_files():
    for base, dirs, names in os.walk(ROOT):
        relative = Path(base).relative_to(ROOT)
        dirs[:] = [
            name
            for name in dirs
            if name not in SKIP_DIRS
            and not (relative == Path("local_deploy") and name == "data")
            and not (relative == Path(".") and name in LOCAL_ROOTS)
        ]
        for name in names:
            yield Path(base) / name


def main() -> int:
    failures: list[str] = []
    for path in iter_files():
        if path.resolve() == Path(__file__).resolve():
            continue
        relative = path.relative_to(ROOT)
        if path.is_symlink():
            resolved = path.resolve(strict=False)
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                failures.append(f"external symlink: {relative} -> {os.readlink(path)}")
            continue
        try:
            if path.stat().st_size > TEXT_LIMIT:
                continue
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        for label, pattern in RULES.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                failures.append(f"{label}: {relative}:{line}")
    if failures:
        print("Public-release check failed:", file=sys.stderr)
        print("\n".join(f"  - {item}" for item in failures), file=sys.stderr)
        return 1
    print("Public-release check passed: no private paths, obvious credentials, or external symlinks found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
