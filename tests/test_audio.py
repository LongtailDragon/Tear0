import sys
import types

import numpy as np

from tear0.audio import (
    WhisperTranscriber,
    _append_tail_silence,
    _frame_rms,
    _is_loud_enough,
    _frames_with_preroll,
    _should_start_speech,
    _split_tts_chunks,
    analyze_speech_frames,
)
from tear0.config import WhisperConfig


def test_frame_rms_treats_silence_as_zero():
    assert _frame_rms(b"\x00\x00" * 480) == 0.0


def test_loud_enough_rejects_low_level_background_noise():
    quiet = (np.ones(480, dtype=np.int16) * 80).tobytes()
    assert _is_loud_enough(quiet, threshold=500) is False


def test_loud_enough_accepts_clear_speech_level_audio():
    speechy = (np.ones(480, dtype=np.int16) * 2500).tobytes()
    assert _is_loud_enough(speechy, threshold=500) is True


def test_should_start_speech_requires_consecutive_vad_frames_and_energy():
    assert _should_start_speech([True, False, True], [3000, 0, 3000], required_frames=2, rms_threshold=500) is False
    assert _should_start_speech([True, True], [100, 100], required_frames=2, rms_threshold=500) is False
    assert _should_start_speech([True, True], [3000, 3000], required_frames=2, rms_threshold=500) is True


def test_append_tail_silence_adds_configured_padding_without_changing_audio_prefix():
    samples = np.array([0.1, -0.2, 0.3], dtype=np.float32)

    padded = _append_tail_silence(samples, sample_rate=1000, tail_padding_ms=250)

    assert np.array_equal(padded[:3], samples)
    assert len(padded) == 253
    assert np.all(padded[3:] == 0.0)


def test_append_tail_silence_noops_when_padding_is_zero():
    samples = np.array([0.1, -0.2, 0.3], dtype=np.float32)

    padded = _append_tail_silence(samples, sample_rate=1000, tail_padding_ms=0)

    assert np.array_equal(padded, samples)


def test_analyze_speech_frames_reports_trigger_and_rms_stats():
    report = analyze_speech_frames([True, True, True], [100, 700, 900], required_frames=2, rms_threshold=500)

    assert report["would_trigger"] is True
    assert report["speech_frames"] == 3
    assert report["peak_rms"] == 900
    assert round(report["average_rms"], 2) == 566.67


def test_frames_with_preroll_keeps_audio_before_trigger_frame():
    preroll = [b"first-word", b"second-word", b"trigger-frame"]

    frames = _frames_with_preroll(preroll, current_frame=b"trigger-frame")

    assert frames == [b"first-word", b"second-word", b"trigger-frame"]


def test_whisper_warm_up_loads_model_before_first_transcription(monkeypatch):
    created = []

    class FakeWhisperModel:
        def __init__(self, model_size, *, device, compute_type, cpu_threads):
            created.append((model_size, device, compute_type, cpu_threads))

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel))
    transcriber = WhisperTranscriber(WhisperConfig(model_size="tiny.en", device="cpu", compute_type="int8", cpu_threads=2))

    transcriber.warm_up()

    assert created == [("tiny.en", "cpu", "int8", 2)]
    assert transcriber._model is not None


def test_split_tts_chunks_keeps_short_text_single_chunk():
    assert _split_tts_chunks("Take the next safe step.", max_chars=50) == ["Take the next safe step."]


def test_split_tts_chunks_splits_long_responses_on_sentence_boundaries():
    chunks = _split_tts_chunks("First sentence. Second sentence is a little longer. Third sentence.", max_chars=35)

    assert chunks == ["First sentence.", "Second sentence is a little longer.", "Third sentence."]


def test_split_tts_chunks_splits_long_sentence_on_words():
    chunks = _split_tts_chunks("one two three four five six", max_chars=13)

    assert chunks == ["one two three", "four five six"]
