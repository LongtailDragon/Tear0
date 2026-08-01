from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WIN_DRIVE = r"C:" + r"/"
WIN_BACKSLASH_DRIVE = r"C:" + r"\\\\"
LOCAL_WEB_ROOT = "wamp" + "64"
KEY_NAMES = ["OPENAI" + "_API_KEY", "ANTHROPIC" + "_API_KEY"]
FORBIDDEN_PATTERNS = [
    re.compile(WIN_BACKSLASH_DRIVE + r"Users\\\\[^\\\s]+", re.IGNORECASE),
    re.compile(WIN_DRIVE + r"Users/[^/\s]+", re.IGNORECASE),
    re.compile(WIN_BACKSLASH_DRIVE + LOCAL_WEB_ROOT, re.IGNORECASE),
    re.compile(WIN_DRIVE + LOCAL_WEB_ROOT, re.IGNORECASE),
    re.compile(r"AppData\\\\Local", re.IGNORECASE),
    re.compile(r"api[_-]?key\s*[:=]\s*['\"][^'\"]+", re.IGNORECASE),
    re.compile(r"(?:secret|token|password)\s*[:=]\s*['\"][^'\"]+", re.IGNORECASE),
    *(re.compile(re.escape(name), re.IGNORECASE) for name in KEY_NAMES),
]

BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".wav", ".mp3", ".onnx", ".bin",
    ".exe", ".dll", ".pyd",
}
GENERATED_PATH_PREFIXES = (
    "tear0.config.json",
    "smoke/",
    ".env",
    ".venv/",
    ".pytest_cache/",
    "dist/",
    "build/",
    "models/",
    "sessions/",
)


def candidate_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        text=True,
    )
    return [ROOT / line for line in output.splitlines() if line]


def is_forbidden_tracked_path(rel: str) -> bool:
    return any(rel == prefix.rstrip("/") or rel.startswith(prefix) for prefix in GENERATED_PATH_PREFIXES)


def scan_tracked_files(paths: list[Path]) -> list[str]:
    failures: list[str] = []
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        if is_forbidden_tracked_path(rel):
            failures.append(f"forbidden tracked artifact: {rel}")
            continue
        if path.suffix.lower() in BINARY_SUFFIXES:
            failures.append(f"tracked generated/binary-like artifact: {rel}")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.search(text):
                failures.append(f"{rel}: matched {pattern.pattern}")
    return failures


def main() -> int:
    failures = scan_tracked_files(candidate_files())
    if failures:
        print("Public-readiness check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Public-readiness check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
