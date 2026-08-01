from pathlib import Path

from tear0.config import HardwareInfo, choose_whisper_config, build_hermes_command, find_nvidia_cuda_dll_dirs, session_dir


def test_choose_whisper_config_prefers_cuda_tiny_en_for_nvidia_gpu():
    hw = HardwareInfo(has_nvidia_gpu=True, gpu_name="Generic NVIDIA GPU", vram_mb=8192)
    cfg = choose_whisper_config(hw, cuda_runtime_available=True)
    assert cfg.device == "cuda"
    assert cfg.compute_type == "float16"
    assert cfg.model_size == "tiny.en"


def test_choose_whisper_config_uses_cpu_when_nvidia_gpu_lacks_cuda_runtime():
    hw = HardwareInfo(has_nvidia_gpu=True, gpu_name="Generic NVIDIA GPU", vram_mb=8192)
    cfg = choose_whisper_config(hw, cuda_runtime_available=False)
    assert cfg.device == "cpu"
    assert cfg.compute_type == "int8"
    assert cfg.model_size == "tiny.en"


def test_choose_whisper_config_uses_cpu_int8_without_gpu():
    hw = HardwareInfo(has_nvidia_gpu=False, gpu_name=None, vram_mb=None)
    cfg = choose_whisper_config(hw)
    assert cfg.device == "cpu"
    assert cfg.compute_type == "int8"
    assert cfg.model_size == "tiny.en"


def test_find_nvidia_cuda_dll_dirs_finds_cublas_and_cudnn_bins(tmp_path):
    site = tmp_path / "site-packages"
    cublas = site / "nvidia" / "cublas" / "bin"
    cudnn = site / "nvidia" / "cudnn" / "bin"
    cublas.mkdir(parents=True)
    cudnn.mkdir(parents=True)
    (cublas / "cublas64_12.dll").write_bytes(b"fake")
    (cudnn / "cudnn64_9.dll").write_bytes(b"fake")

    dirs = find_nvidia_cuda_dll_dirs([str(site)])

    assert cublas in dirs
    assert cudnn in dirs


def test_build_hermes_command_attaches_prompt_and_image_without_shell_joining(tmp_path):
    image = tmp_path / "shot.png"
    image.write_bytes(b"fake")
    cmd = build_hermes_command("hello \"world\"", image, max_turns=4)
    assert cmd[:2] == ["hermes", "chat"]
    assert "--quiet" not in cmd
    assert "--image" in cmd
    assert str(image) in cmd
    assert "hello \"world\"" in cmd
    assert "--max-turns" in cmd
    assert "4" in cmd


def test_build_hermes_command_can_continue_named_session(tmp_path):
    image = tmp_path / "shot.png"
    image.write_bytes(b"fake")

    cmd = build_hermes_command("remember this", image, session_name="tear0-live")

    assert cmd[:4] == ["hermes", "--continue", "tear0-live", "chat"]
    assert "--image" in cmd
    assert "remember this" in cmd


def test_build_hermes_command_can_resume_session_id(tmp_path):
    image = tmp_path / "shot.png"
    image.write_bytes(b"fake")

    cmd = build_hermes_command("next step", image, resume_session_id="20260728_abc123")

    assert cmd[:4] == ["hermes", "--resume", "20260728_abc123", "chat"]
    assert "next step" in cmd


def test_session_dir_is_inside_base_and_named_session(tmp_path):
    base = tmp_path / "tear0-test"
    path = session_dir(base, "abc123")
    assert path == base / "abc123"
