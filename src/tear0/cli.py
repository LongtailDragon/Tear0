from __future__ import annotations

import argparse
from collections.abc import Callable
import re
import subprocess
import sys
import time
from pathlib import Path

from .audio import KokoroSpeaker, SpeechRecorder, WhisperTranscriber
from .config import build_hermes_command, clear_session_cache, config_path, create_default_config, load_config, save_config, session_dir
from .hotkeys import PauseToggle, WindowsCtrlPlusHotkey
from .output import format_ascii_box, print_bullet
from .vision import capture_display, prompt_for_display

CONNECTED = "Tear0 connected. How can I help?"
GOODBYE = "Tear0 disconnected."
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def classify_control_command(text: str) -> str | None:
    normalized = " ".join(text.lower().replace("tear zero", "tear0").split())
    commands = {
        "quit": {"quit", "exit", "stop tear0", "goodbye"},
    }
    for action, aliases in commands.items():
        if normalized in aliases:
            return action
    return None


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_hermes_stdout(stdout: str) -> tuple[str | None, str]:
    session_id = None
    answer_lines = []
    for line in stdout.splitlines():
        if line.startswith("session_id:"):
            session_id = line.split(":", 1)[1].strip() or None
        else:
            answer_lines.append(line)
    return session_id, "\n".join(answer_lines).strip()


def parse_hermes_cli_output(output: str) -> tuple[str | None, str]:
    session_id = None
    answer_lines: list[str] = []
    in_answer_box = False
    for line in output.splitlines():
        clean_line = strip_ansi(line)
        stripped = clean_line.strip()
        if stripped.startswith("Session:"):
            session_id = stripped.split(":", 1)[1].strip() or None
        if stripped.startswith("session_id:"):
            session_id = stripped.split(":", 1)[1].strip() or None
            continue
        if stripped.startswith("╭") and "Hermes" in stripped:
            in_answer_box = True
            answer_lines = []
            continue
        if in_answer_box:
            if stripped.startswith("╰"):
                in_answer_box = False
                continue
            answer_lines.append(clean_line[4:] if clean_line.startswith("    ") else stripped)
    if answer_lines:
        return session_id, "\n".join(answer_lines).strip()
    return parse_hermes_stdout(output)


def should_stream_hermes_progress_line(line: str) -> bool:
    stripped = strip_ansi(line).strip()
    if not stripped:
        return False
    hidden_prefixes = (
        "Query:",
        "Initializing agent",
        "📎 attaching ",
        "User voice command:",
        "Resume this session with:",
        "hermes --resume ",
        "Session:",
        "Duration:",
        "Messages:",
        "session_id:",
    )
    if any(stripped.startswith(prefix) for prefix in hidden_prefixes):
        return False
    if set(stripped) <= {"─", "-", "━", "="}:
        return False
    return is_hermes_activity_line(stripped)


def is_hermes_activity_line(line: str) -> bool:
    lowered = line.lower()
    thinking_words = ("mulling", "thinking", "pondering", "reasoning", "working", "planning")
    if any(word in lowered for word in thinking_words):
        return True
    activity_words = (
        "tool",
        "command",
        "terminal",
        "running",
        "executing",
        "reading",
        "searching",
        "writing",
        "patching",
    )
    if any(word in lowered for word in activity_words):
        return True
    tool_icons = ("🔧", "📋", "🔎", "🔍", "📖", "📄", "📁", "📝", "💻", "🖥", "⚙", "🧰", "🌐", "🧪", "📚", "✅")
    if line.startswith(tool_icons):
        return True
    return bool(re.search(r"\b\d+(?:\.\d+)?s$", line))


def truncate_progress_line(line: str, *, max_chars: int = 100) -> str:
    line = " ".join(strip_ansi(line).split())
    if len(line) <= max_chars:
        return line
    return line[: max_chars - 3].rstrip() + "..."


def parse_hermes_session_id(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("session_id:"):
            return line.split(":", 1)[1].strip() or None
    return None


def hermes_ask(
    prompt: str,
    image_path: Path,
    *,
    hermes_executable: str,
    max_turns: int,
    session_name: str | None = None,
    resume_session_id: str | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[str, str | None]:
    command = build_hermes_command(
        prompt,
        image_path,
        max_turns=max_turns,
        hermes_executable=hermes_executable,
        session_name=session_name,
        resume_session_id=resume_session_id,
    )
    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    output_lines: list[str] = []
    in_answer_box = False
    for line in process.stdout:
        clean_line = line.rstrip("\r\n")
        output_lines.append(clean_line)
        progress_line = strip_ansi(clean_line)
        stripped = progress_line.strip()
        if stripped.startswith("╭") and "Hermes" in stripped:
            in_answer_box = True
            continue
        if in_answer_box:
            if stripped.startswith("╰"):
                in_answer_box = False
            continue
        if on_progress and should_stream_hermes_progress_line(progress_line):
            on_progress(truncate_progress_line(progress_line))
    return_code = process.wait()
    output = "\n".join(output_lines)
    if return_code != 0:
        raise RuntimeError(f"Hermes failed with exit code {return_code}: {output.strip()}")
    stdout_session_id, answer = parse_hermes_cli_output(output)
    new_session_id = stdout_session_id or parse_hermes_session_id(output)
    return answer, new_session_id


def run_loop(args: argparse.Namespace) -> int:
    root = project_root()
    cfg_file = Path(args.config) if args.config else config_path(root)
    if cfg_file.exists():
        cfg = load_config(cfg_file)
    else:
        cfg = create_default_config()
        save_config(cfg_file, cfg)

    if args.dry_run:
        print_bullet("Tear0 dry run")
        print_bullet("Config:", cfg_file)
        print_bullet("Hardware:", cfg.hardware)
        print_bullet("Whisper:", cfg.whisper)
        print_bullet("Session root:", cfg.session_root)
        print_bullet("Hermes executable:", cfg.hermes_executable)
        print_bullet("Hermes continuity:", "per Tear0 run; first turn creates a session, later turns use --resume")
        print_bullet("Kokoro model:", cfg.kokoro_model_path)
        return 0

    cache_root = Path(cfg.session_root)
    if args.audio_test:
        sess = session_dir(cache_root, "audio-test")
        sess.mkdir(parents=True, exist_ok=True)
        recorder = SpeechRecorder(cfg)
        audio_path = sess / "probe.wav"
        print_bullet(f"Audio test: speak normally for {args.audio_test:.1f} seconds...")
        report = recorder.record_probe(audio_path, seconds=args.audio_test)
        print_bullet("Input device:", report["input_device"])
        print_bullet("Recorded:", report["path"])
        print_bullet("Frames:", report["frames"])
        print_bullet("VAD speech frames:", report["speech_frames"])
        print_bullet("Peak RMS:", round(report["peak_rms"], 1))
        print_bullet("Average RMS:", round(report["average_rms"], 1))
        print_bullet("Would trigger Tear0:", "YES" if report["would_trigger"] else "NO")
        print_bullet("Transcribing probe...")
        text = WhisperTranscriber(cfg.whisper).transcribe(audio_path)
        print_bullet("Transcript:", text or "<empty>")
        return 0

    clear_session_cache(cache_root)
    display = prompt_for_display()
    sess = session_dir(cache_root)
    sess.mkdir(parents=True, exist_ok=True)

    speaker = KokoroSpeaker(cfg)
    recorder = SpeechRecorder(cfg)
    transcriber = WhisperTranscriber(cfg.whisper)
    pause_toggle = PauseToggle(
        on_pause=lambda: print(format_ascii_box("Tear0", "Paused. Press Ctrl+= or Ctrl++ to resume listening.")),
        on_resume=lambda: print(format_ascii_box("Tear0", "Resumed. Listening again.")),
    )
    pause_hotkey = WindowsCtrlPlusHotkey(pause_toggle)
    hotkey_active = pause_hotkey.start()
    if hotkey_active:
        print_bullet("Pause toggle: Ctrl+= or Ctrl++")
    else:
        print_bullet("Pause hotkey is only available on Windows.")
    print_bullet("Loading Whisper model...")
    transcriber.warm_up()
    print_bullet("Loading Kokoro voice model...")
    speaker.warm_up()
    try:
        print(format_ascii_box("Tear0", CONNECTED))
        speaker.speak(CONNECTED)
        turn = 0
        hermes_session_id: str | None = None
        muted = False
        while True:
            turn += 1
            audio_path = sess / f"turn-{turn:04d}.wav"
            screenshot_path = sess / f"turn-{turn:04d}.png"

            pause_toggle.wait_if_paused()

            def snap():
                capture_display(display.index, screenshot_path)
                print_bullet("Screenshot captured:", screenshot_path)

            recorded_path = recorder.record_next_utterance(audio_path, on_speech_start=snap, should_pause=pause_toggle.paused.is_set)
            if recorded_path is None:
                continue
            print_bullet("Transcribing...")
            text = transcriber.transcribe(audio_path)
            if not text:
                print_bullet("No transcription detected; listening again.")
                continue
            print(format_ascii_box("You", text))

            control = classify_control_command(text)
            if control == "quit":
                break

            if not screenshot_path.exists():
                capture_display(display.index, screenshot_path)
            prompt = (
                "You are Tear0, a concise live desktop support agent. "
                "The attached screenshot is the user's selected display captured at the moment speech began. "
                "Answer conversationally. Keep your responses practical and **extremely** concise. "
                "Answer in as little text as possible. "
                "Keep your responses entirely vocable; Do not include in your response anything that would not normally be said aloud to a human, such as complete file paths, asemantic tokens or hashes, or anything else that doesn't make sense if said aloud. "
                "If the user needs to take an action, give the next safest actionable step. "
                "Do not include the phrase 'next safest step' or 'next actionable step' in your response.\n\n"
                f"User voice command: {text}"
            )
            print_bullet("Sending to Hermes...")
            answer, new_session_id = hermes_ask(
                prompt,
                screenshot_path,
                hermes_executable=cfg.hermes_executable,
                max_turns=args.max_turns,
                resume_session_id=hermes_session_id,
                on_progress=print_bullet,
            )
            if new_session_id and new_session_id != hermes_session_id:
                hermes_session_id = new_session_id
                print_bullet("Hermes session id:", hermes_session_id)
            print(format_ascii_box("Hermes", answer))
            if not muted:
                speaker.speak(answer)
    except KeyboardInterrupt:
        print_bullet("Stopping Tear0...")
    finally:
        try:
            if "muted" not in locals() or not muted:
                speaker.speak(GOODBYE)
        except Exception:
            pass
        pause_hotkey.stop()
        clear_session_cache(cache_root)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tear0 voice + vision turn loop for Hermes Agent")
    parser.add_argument("--config", help="Path to tear0.config.json")
    parser.add_argument("--dry-run", action="store_true", help="Inspect config/hardware without opening microphone")
    parser.add_argument("--audio-test", type=float, metavar="SECONDS", help="Record a short microphone diagnostic clip and print VAD/RMS/transcript results")
    parser.add_argument("--max-turns", type=int, default=20, help="Hermes max tool turns per request")
    args = parser.parse_args(argv)
    return run_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
