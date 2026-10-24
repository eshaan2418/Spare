"""Repository hygiene scan: secrets, large files, raw personal photos (run in CI and before release)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

SECRET_PATTERNS = [
    r"AKIA[0-9A-Z]{16}", r"sk-[A-Za-z0-9]{32,}", r"sk-ant-[A-Za-z0-9\-_]{20,}", r"hf_[A-Za-z0-9]{30,}",
    r"ghp_[A-Za-z0-9]{36}", r"-----BEGIN (RSA|OPENSSH|EC) PRIVATE KEY-----", r"xox[baprs]-[A-Za-z0-9-]{10,}",
]
PHOTO_EXT = {".jpg", ".jpeg", ".png", ".heic", ".JPG", ".JPEG", ".PNG", ".HEIC"}
MAX_BYTES = 5_000_000


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=False).stdout.split()
    return [Path(p) for p in out]


def history_blobs_over(limit: int) -> list[str]:
    cmd = "git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)'"
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False).stdout
    big = []
    for line in out.splitlines():
        parts = line.split(" ", 3)
        if len(parts) >= 3 and parts[0] == "blob" and int(parts[2]) > limit:
            big.append(line)
    return big


def main() -> int:
    problems = []
    for p in tracked_files():
        if not p.exists():
            continue
        if p.suffix in PHOTO_EXT and p.parts[:2] == ("data", "real"):
            problems.append(f"raw photo tracked: {p}")
        if p.stat().st_size > MAX_BYTES:
            problems.append(f"large file tracked: {p} ({p.stat().st_size} B)")
        if p.suffix in {".py", ".md", ".txt", ".yml", ".yaml", ".json", ".toml", ".ini", ".sh", ".cfg", ""}:
            try:
                txt = p.read_text(errors="ignore")
            except Exception:  # noqa: BLE001
                continue
            for pat in SECRET_PATTERNS:
                if re.search(pat, txt):
                    problems.append(f"possible secret ({pat}) in {p}")
    problems += [f"large blob in history: {b}" for b in history_blobs_over(MAX_BYTES)]
    log = subprocess.run(["git", "log", "--all", "-p", "--no-color"], capture_output=True, text=True, check=False).stdout
    for pat in SECRET_PATTERNS:
        if re.search(pat, log):
            problems.append(f"possible secret in git history ({pat})")
    for pr in problems:
        print("SCAN:", pr)
    print(f"repo_scan: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
