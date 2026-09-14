#!/usr/bin/env python3
"""Regenerate MANIFEST.sha256 for the review archive."""
from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXCLUDE = {"MANIFEST.sha256", ".DS_Store", "outputs/reproduced_results.json"}


def main() -> None:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if not path.is_file() or rel.as_posix() in EXCLUDE or rel.name in EXCLUDE or any(part in {".git", ".venv", "__pycache__"} for part in rel.parts):
            continue
        rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  ./{rel.as_posix()}")
    (ROOT / "MANIFEST.sha256").write_text("\n".join(rows) + "\n")
    print(f"wrote {len(rows)} digests")


if __name__ == "__main__":
    main()
