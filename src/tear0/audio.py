from __future__ import annotations

import queue
import re
import sys
import wave
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad

from .config import Tear0Config, WhisperConfig, activate_packaged_cuda_dll_dirs
from .output import print_bullet


def _frame_rms(pcm16: bytes) -> float:
    if not pcm16:
        return 0.0
    samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples * samples)))


def _is_loud_enough(pcm16: bytes, threshold: int) -> bool:
    return _frame_rms(pcm16) >= threshold


def _should_start_speech(vad_hits, rms_values, *, required_frames: int, rms_threshold: int) -> bool:
    if len(vad_hits) < required_frames or len(rms_values) < required_frames:
        return False
    recent_vad = list(vad_hits)[-required_frames:]
    recent_rms = list(rms_values)[-required_frames:]
    return all(recent_vad) and all(value >= rms_threshold for value in recent_rms)


def analyze_speech_frames(vad_hits, rms_values, *, required_frames: int, rms_threshold: int) -> dict:
    rms_values = list(rms_values)
    vad_hits = list(vad_hits)
    return {
        "frames": len(rms_values),
        "speech_frames": sum(1 for hit in vad_hits if hit),
        "peak_rms": max(rms_values) if rms_values else 0.0,
        "average_rms": float(sum(rms_values) / len(rms_values)) if rms_values else 0.0,
        "would_trigger": _should_start_speech(
            vad_hits,
            rms_values,
            required_frames=required_frames,
            rms_threshold=rms_threshold,
        ),
    }


def _frames_with_preroll(preroll_frames, *, current_frame: bytes) -> list[bytes]:
    """Return buffered audio leading into the speech trigger without duplicates."""
    frames = list(preroll_frames)
    if not frames or frames[-1] != current_frame:
        frames.append(current_frame)
    return frames


def _append_tail_silence(samples, *, sample_rate: int, tail_padding_ms: int):
    """Append silence so Windows audio devices do not clip the final phoneme."""
    if tail_padding_ms <= 0:
        return samples
    padding_samples = int(sample_rate * tail_padding_ms / 1000)
    if padding_samples <= 0:
        return samples
    tail_shape = (padding_samples,) + tuple(samples.shape[1:])
    tail = np.zeros(tail_shape, dtype=samples.dtype)
    return np.concatenate([samples, tail], axis=0)


def _split_tts_chunks(text: str, *, max_chars: int = 220) -> list[str]:
    """Split text so Kokoro can start speaking before a long full response is synthesized."""
    text = " ".join(text.split())
    if not text:
        return []
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_tts_sentence(sentence, max_chars=max_chars))
            continue
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _split_long_tts_sentence(sentence: str, *, max_chars: int) -> list[str]:
    words = sentence.split()
    chunks: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def write_wav(path: Path, pcm16: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16)


class SpeechRecorder:
    def __init__(self, cfg: Tear0Config):
        self.sample_rate = cfg.sample_rate
        self.vad = webrtcvad.Vad(cfg.vad_aggressiveness)
        self.silence_frames = max(1, cfg.silence_ms // 30)
        self.max_frames = max(1, int(cfg.max_utterance_seconds * 1000 / 30))
        self.frame_ms = 30
        self.frame_samples = int(self.sample_rate * self.frame_ms / 1000)
        self.speech_start_frames = max(1, cfg.speech_start_frames)
        self.speech_rms_threshold = max(0, cfg.speech_rms_threshold)
        self.pre_speech_frames = max(self.speech_start_frames, int(cfg.pre_speech_ms / self.frame_ms))

    def input_device_label(self) -> str:
        try:
            device = sd.query_devices(kind="input")
            return f"{device['name']}"
        except Exception as exc:
            return f"unknown input device ({exc})"

    def record_probe(self, out_path: Path, *, seconds: float = 5.0) -> dict:
        """Record a fixed-duration clip and return VAD/RMS diagnostics."""
        seconds = max(1.0, float(seconds))
        q: queue.Queue[bytes] = queue.Queue()
        frames: list[bytes] = []
        vad_hits: list[bool] = []
        rms_values: list[float] = []
        target_frames = max(1, int(seconds * 1000 / self.frame_ms))

        def callback(indata, frames_count, time_info, status):
            if status:
                print_bullet(f"Audio warning: {status}", file=sys.stderr)
            pcm = np.clip(indata[:, 0], -1.0, 1.0)
            q.put((pcm * 32767).astype(np.int16).tobytes())

        with sd.InputStream(channels=1, samplerate=self.sample_rate, blocksize=self.frame_samples, dtype="float32", callback=callback):
            for _ in range(target_frames):
                frame = q.get()
                frames.append(frame)
                vad_hits.append(self.vad.is_speech(frame, self.sample_rate))
                rms_values.append(_frame_rms(frame))

        pcm16 = b"".join(frames)
        write_wav(out_path, pcm16, self.sample_rate)
        report = analyze_speech_frames(
            vad_hits,
            rms_values,
            required_frames=self.speech_start_frames,
            rms_threshold=self.speech_rms_threshold,
        )
        report["path"] = str(out_path)
        report["seconds"] = seconds
        report["input_device"] = self.input_device_label()
        return report

    def record_next_utterance(self, out_path: Path, on_speech_start=None, should_pause: Callable[[], bool] | None = None) -> Path | None:
        """Block until speech starts, then record until trailing silence."""
        if should_pause and should_pause():
            return None
        q: queue.Queue[bytes] = queue.Queue()
        started = False
        frames: list[bytes] = []
        silence = 0
        vad_window = deque(maxlen=self.speech_start_frames)
        rms_window = deque(maxlen=self.speech_start_frames)
        preroll = deque(maxlen=self.pre_speech_frames)

        def callback(indata, frames_count, time_info, status):
            if status:
                print_bullet(f"Audio warning: {status}", file=sys.stderr)
            pcm = np.clip(indata[:, 0], -1.0, 1.0)
            q.put((pcm * 32767).astype(np.int16).tobytes())

        with sd.InputStream(channels=1, samplerate=self.sample_rate, blocksize=self.frame_samples, dtype="float32", callback=callback):
            print_bullet(f"Listening on {self.input_device_label()}... speak when ready. Press Ctrl+C to stop.")
            print_bullet(f"Speech gate: {self.speech_start_frames} consecutive speech frames, RMS >= {self.speech_rms_threshold}.")
            while True:
                if should_pause and should_pause():
                    return None
                frame = q.get()
                if should_pause and should_pause():
                    return None
                is_speech = self.vad.is_speech(frame, self.sample_rate)
                rms = _frame_rms(frame)
                preroll.append(frame)
                vad_window.append(is_speech)
                rms_window.append(rms)
                just_started = False
                if not started and _should_start_speech(
                    vad_window,
                    rms_window,
                    required_frames=self.speech_start_frames,
                    rms_threshold=self.speech_rms_threshold,
                ):
                    started = True
                    just_started = True
                    frames = _frames_with_preroll(preroll, current_frame=frame)
                    if on_speech_start:
                        on_speech_start()
                    print_bullet(f"Speech detected; recording command... included {len(frames)} pre-roll frames.")
                if started:
                    if not just_started:
                        frames.append(frame)
                    silence = 0 if is_speech else silence + 1
                    if silence >= self.silence_frames or len(frames) >= self.max_frames:
                        break
        pcm16 = b"".join(frames)
        write_wav(out_path, pcm16, self.sample_rate)
        return out_path


class WhisperTranscriber:
    def __init__(self, cfg: WhisperConfig):
        self.cfg = cfg
        self._model = None

    def _fallback_to_cpu(self, exc: Exception):
        print_bullet(f"CUDA Whisper failed ({exc}); falling back to CPU int8.", file=sys.stderr)
        self.cfg = WhisperConfig(model_size=self.cfg.model_size, device="cpu", compute_type="int8", cpu_threads=self.cfg.cpu_threads)
        self._model = None
        return self._load_model()

    def _load_model(self):
        if self._model is None:
            if self.cfg.device == "cuda":
                activate_packaged_cuda_dll_dirs()
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.cfg.model_size,
                device=self.cfg.device,
                compute_type=self.cfg.compute_type,
                cpu_threads=self.cfg.cpu_threads,
            )
        return self._model

    def warm_up(self) -> None:
        try:
            self._load_model()
        except Exception as exc:
            if self.cfg.device != "cuda":
                raise
            self._fallback_to_cpu(exc)

    def transcribe(self, audio_path: Path) -> str:
        try:
            segments, _ = self._load_model().transcribe(
                str(audio_path),
                language=self.cfg.language,
                beam_size=self.cfg.beam_size,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            # faster-whisper returns a generator; CUDA/library errors can happen
            # during iteration, not only during the transcribe() call.
            return " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as exc:
            if self.cfg.device != "cuda":
                raise
            segments, _ = self._fallback_to_cpu(exc).transcribe(
                str(audio_path), language=self.cfg.language, beam_size=1, vad_filter=True, condition_on_previous_text=False
            )
            return " ".join(seg.text.strip() for seg in segments).strip()


class KokoroSpeaker:
    def __init__(self, cfg: Tear0Config):
        self.model_path = Path(cfg.kokoro_model_path)
        self.voices_path = Path(cfg.kokoro_voices_path)
        self.voice = cfg.kokoro_voice
        self.tail_padding_ms = max(0, cfg.tts_tail_padding_ms)
        self._kokoro = None

    def _load(self):
        if self._kokoro is None:
            from kokoro_onnx import Kokoro

            if not self.model_path.exists() or not self.voices_path.exists():
                raise FileNotFoundError(
                    "Kokoro model files are missing. Run install.ps1 or python -m tear0.installer first."
                )
            self._kokoro = Kokoro(str(self.model_path), str(self.voices_path))
        return self._kokoro

    def warm_up(self) -> None:
        self._load()

    def _synthesize(self, text: str, *, final_chunk: bool = True):
        kokoro = self._load()
        samples, sample_rate = kokoro.create(text, voice=self.voice, speed=1.0, lang="en-us")
        if final_chunk:
            samples = _append_tail_silence(samples, sample_rate=sample_rate, tail_padding_ms=self.tail_padding_ms)
        return samples, sample_rate

    def synthesize_to_file(self, text: str, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        samples, sample_rate = self._synthesize(text, final_chunk=True)
        sf.write(str(out_path), samples, sample_rate)
        return out_path

    def speak(self, text: str) -> None:
        chunks = _split_tts_chunks(text)
        if not chunks:
            return
        sd.stop()
        if len(chunks) == 1:
            samples, sample_rate = self._synthesize(chunks[0], final_chunk=True)
            sd.play(samples, sample_rate)
            sd.wait()
            return

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._synthesize, chunks[0], final_chunk=False)
            for index in range(len(chunks)):
                samples, sample_rate = future.result()
                next_index = index + 1
                if next_index < len(chunks):
                    future = executor.submit(self._synthesize, chunks[next_index], final_chunk=next_index == len(chunks) - 1)
                sd.play(samples, sample_rate)
                sd.wait()
