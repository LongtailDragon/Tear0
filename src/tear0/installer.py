from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import requests

from .config import config_path, create_default_config, save_config
from .output import print_bullet

MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1024:
        print_bullet(f"Already present: {dest}")
        return
    print_bullet(f"Downloading {url} -> {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(dest)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print_bullet("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def install(project_root: Path, *, skip_models: bool = False) -> None:
    cfg = create_default_config()
    save_config(config_path(project_root), cfg)
    print_bullet("Wrote config:", config_path(project_root))
    print_bullet("Hardware:", cfg.hardware)
    print_bullet("Whisper:", cfg.whisper)
    if not skip_models:
        download(MODEL_URL, Path(cfg.kokoro_model_path))
        download(VOICES_URL, Path(cfg.kokoro_voices_path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install/configure Tear0 assets")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--skip-models", action="store_true")
    args = parser.parse_args(argv)
    install(Path(args.project_root), skip_models=args.skip_models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
