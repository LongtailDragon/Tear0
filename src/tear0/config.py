from __future__ import annotations

import json
import os
import platform
import shutil
import site
import subprocess
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


APP_NAME = "Tear0"


@dataclass(frozen=True)
class HardwareInfo:
    has_nvidia_gpu: bool
    gpu_name: Optional[str] = None
    vram_mb: Optional[int] = None
    cuda_available: bool = False


@dataclass(frozen=True)
class WhisperConfig:
    model_size: str
    device: str
    compute_type: str
    cpu_threads: int = 4
    beam_size: int = 1
    language: str = "en"


@dataclass(frozen=True)
class Tear0Config:
    hardware: HardwareInfo
    whisper: WhisperConfig
    hermes_executable: str = "hermes"
    session_root: str = ""
    kokoro_model_path: str = ""
    kokoro_voices_path: str = ""
    kokoro_voice: str = "af_heart"
    tts_tail_padding_ms: int = 400
    sample_rate: int = 16000
    vad_aggressiveness: int = 2
    silence_ms: int = 700
    speech_start_frames: int = 3
    speech_rms_threshold: int = 500
    pre_speech_ms: int = 450
    max_utterance_seconds: int = 30


def inspect_hardware() -> HardwareInfo:
    """Inspect host hardware without requiring privileged APIs."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=5,
        ).strip()
    except Exception:
        return HardwareInfo(has_nvidia_gpu=False)

    if not out:
        return HardwareInfo(has_nvidia_gpu=False)
    first = out.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    name = parts[0] if parts else None
    vram_mb = None
    if len(parts) > 1:
        try:
            vram_mb = int(float(parts[1]))
        except ValueError:
            vram_mb = None
    return HardwareInfo(has_nvidia_gpu=True, gpu_name=name, vram_mb=vram_mb, cuda_available=True)


def _dll_exists_on_path(dll_name: str) -> bool:
    for item in os.environ.get("PATH", "").split(os.pathsep):
        if item and (Path(item) / dll_name).exists():
            return True
    return False


def find_nvidia_cuda_dll_dirs(site_packages: list[str] | None = None) -> list[Path]:
    site_packages = site_packages or site.getsitepackages()
    dll_dirs: list[Path] = []
    for base in site_packages:
        root = Path(base) / "nvidia"
        for subdir, marker in [("cublas/bin", "cublas64_12.dll"), ("cudnn/bin", "cudnn64_9.dll")]:
            candidate = root / subdir
            if (candidate / marker).exists() and candidate not in dll_dirs:
                dll_dirs.append(candidate)
    return dll_dirs


def activate_packaged_cuda_dll_dirs() -> list[Path]:
    dll_dirs = find_nvidia_cuda_dll_dirs()
    for dll_dir in dll_dirs:
        path_str = str(dll_dir)
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(path_str)
        if path_str not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = path_str + os.pathsep + os.environ.get("PATH", "")
    return dll_dirs


def whisper_cuda_runtime_available() -> bool:
    """Check for the CUDA runtime DLLs faster-whisper/ctranslate2 needs.

    `nvidia-smi` can work even when the CUDA/cuBLAS runtime DLLs are missing.
    If we choose CUDA in that state, the first transcription pays a slow failed
    GPU initialization before falling back to CPU. Avoid that latency up front.
    """
    if platform.system() == "Windows":
        activate_packaged_cuda_dll_dirs()
        return _dll_exists_on_path("cublas64_12.dll") and _dll_exists_on_path("cudnn64_9.dll")
    return True


def choose_whisper_config(hw: HardwareInfo, *, cuda_runtime_available: bool | None = None) -> WhisperConfig:
    """Choose the lowest-latency Whisper settings for turn-based commands."""
    # tiny.en is intentionally used even on fast GPUs: command latency matters more
    # than long-form transcription quality for this app.
    if cuda_runtime_available is None:
        cuda_runtime_available = whisper_cuda_runtime_available()
    if hw.has_nvidia_gpu and cuda_runtime_available:
        return WhisperConfig(model_size="tiny.en", device="cuda", compute_type="float16")
    return WhisperConfig(model_size="tiny.en", device="cpu", compute_type="int8", cpu_threads=max(1, min((os.cpu_count() or 4), 8)))


def default_session_root() -> Path:
    return Path(tempfile.gettempdir()) / "Tear0" / "sessions"


def default_model_dir() -> Path:
    return Path.home() / ".tear0" / "models"


def session_dir(base: Path, session_id: str | None = None) -> Path:
    return base / (session_id or uuid.uuid4().hex)


def clear_session_cache(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)


def build_hermes_command(
    prompt: str,
    image_path: Path,
    *,
    max_turns: int = 20,
    hermes_executable: str = "hermes",
    session_name: str | None = None,
    resume_session_id: str | None = None,
) -> list[str]:
    command = [hermes_executable]
    if resume_session_id:
        command.extend(["--resume", resume_session_id])
    elif session_name:
        command.extend(["--continue", session_name])
    command.extend([
        "chat",
        "--source",
        "tear0",
        "--max-turns",
        str(max_turns),
        "--image",
        str(image_path),
        "--query",
        prompt,
    ])
    return command


def create_default_config() -> Tear0Config:
    hw = inspect_hardware()
    model_dir = default_model_dir()
    return Tear0Config(
        hardware=hw,
        whisper=choose_whisper_config(hw),
        session_root=str(default_session_root()),
        kokoro_model_path=str(model_dir / "kokoro-v1.0.onnx"),
        kokoro_voices_path=str(model_dir / "voices-v1.0.bin"),
    )


def save_config(path: Path, cfg: Tear0Config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")


def load_config(path: Path) -> Tear0Config:
    data = json.loads(path.read_text(encoding="utf-8"))
    hw = HardwareInfo(**data["hardware"])
    whisper = WhisperConfig(**data["whisper"])
    data.setdefault("speech_start_frames", 3)
    data.setdefault("speech_rms_threshold", 500)
    data.setdefault("pre_speech_ms", 450)
    data.setdefault("tts_tail_padding_ms", 400)
    return Tear0Config(**{**data, "hardware": hw, "whisper": whisper})


def config_path(project_root: Path) -> Path:
    return project_root / "tear0.config.json"
