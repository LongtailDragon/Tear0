from tear0.cli import (
    classify_control_command,
    format_ascii_box,
    parse_hermes_cli_output,
    parse_hermes_session_id,
    parse_hermes_stdout,
    print_bullet,
    should_stream_hermes_progress_line,
    strip_ansi,
    truncate_progress_line,
)


def test_parse_hermes_stdout_extracts_session_id_and_answer():
    session_id, answer = parse_hermes_stdout("session_id: 20260728_abc123\nOK\n")

    assert session_id == "20260728_abc123"
    assert answer == "OK"


def test_parse_hermes_stdout_handles_answer_without_session_id():
    session_id, answer = parse_hermes_stdout("Just the answer\n")

    assert session_id is None
    assert answer == "Just the answer"


def test_parse_hermes_session_id_extracts_id_from_stderr():
    assert parse_hermes_session_id("↻ Resumed session abc\n\nsession_id: 20260728_abc123\n") == "20260728_abc123"


def test_parse_hermes_cli_output_extracts_boxed_answer_and_session():
    output = """Query: hidden prompt
Initializing agent...
────────────────────────────────────────

╭─ ⚕ Hermes ───────────────────────────────────────────────────────────────────╮
    OK, do this.
    Then this.
╰──────────────────────────────────────────────────────────────────────────────╯

Resume this session with:
  hermes --resume 20260801_abc123

Session:        20260801_abc123
Duration:       5s
"""

    session_id, answer = parse_hermes_cli_output(output)

    assert session_id == "20260801_abc123"
    assert answer == "OK, do this.\nThen this."


def test_parse_hermes_cli_output_handles_ansi_colored_boxes():
    output = "\x1b[1m╭─ ⚕ Hermes ─╮\x1b[0m\n    \x1b[32mok\x1b[0m\n╰────╯\nSession: 20260801_abc123"

    session_id, answer = parse_hermes_cli_output(output)

    assert session_id == "20260801_abc123"
    assert answer == "ok"


def test_should_stream_hermes_progress_line_filters_prompt_answer_metadata():
    assert should_stream_hermes_progress_line("mulling...") is True
    assert should_stream_hermes_progress_line("Running terminal command...") is True
    assert should_stream_hermes_progress_line("📋 plan 2 task(s) 0.0s") is True
    assert should_stream_hermes_progress_line("🔎 grep (repo/api[_-]?key|secret) 2.1s") is True
    assert should_stream_hermes_progress_line("Query: hidden prompt") is False
    assert should_stream_hermes_progress_line("Initializing agent...") is False
    assert should_stream_hermes_progress_line("  📎 attaching 1 image(s) natively") is False
    assert should_stream_hermes_progress_line("────────────────────────────────────────") is False
    assert should_stream_hermes_progress_line("screenshot is the user's selected display captured at speech start") is False
    assert should_stream_hermes_progress_line("Answer conversationally. Be concise and practical.") is False
    assert should_stream_hermes_progress_line("User voice command: make the repo public") is False
    assert should_stream_hermes_progress_line("Session: 20260801_abc123") is False
    assert should_stream_hermes_progress_line("") is False


def test_strip_ansi_removes_color_codes():
    assert strip_ansi("\x1b[32mok\x1b[0m") == "ok"


def test_truncate_progress_line_caps_long_messages_with_ellipsis():
    line = "x" * 120

    truncated = truncate_progress_line(line, max_chars=100)

    assert len(truncated) == 100
    assert truncated.endswith("...")


def test_format_ascii_box_wraps_label_and_multiline_text():
    box = format_ascii_box("You", "first line\nsecond line", width=24)

    assert box.splitlines() == [
        "",
        "+----------------------+",
        "| You                  |",
        "+----------------------+",
        "| first line           |",
        "| second line          |",
        "+----------------------+",
    ]
    assert box.endswith("\n")


def test_print_bullet_prefixes_non_box_lines(capsys):
    print_bullet("Sending to Hermes...")

    assert capsys.readouterr().out == "> Sending to Hermes...\n"


def test_classify_control_command_detects_quit_only():
    assert classify_control_command("quit") == "quit"
    assert classify_control_command("mute") is None
    assert classify_control_command("unmute Tear0") is None
    assert classify_control_command("pause tear0") is None
    assert classify_control_command("open the browser") is None
