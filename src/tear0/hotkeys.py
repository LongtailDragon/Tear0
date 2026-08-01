from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable


class PauseToggle:
    def __init__(self, *, on_pause: Callable[[], None], on_resume: Callable[[], None]):
        self.paused = threading.Event()
        self._on_pause = on_pause
        self._on_resume = on_resume

    def toggle(self) -> bool:
        if self.paused.is_set():
            self.paused.clear()
            self._on_resume()
            return False
        self.paused.set()
        self._on_pause()
        return True

    def wait_if_paused(self, *, poll_seconds: float = 0.1) -> None:
        while self.paused.is_set():
            time.sleep(poll_seconds)


class WindowsCtrlPlusHotkey:
    """Poll Ctrl+= / Ctrl++ without adding a global keyboard-hook dependency."""

    VK_CONTROL = 0x11
    VK_OEM_PLUS = 0xBB

    def __init__(self, pause_toggle: PauseToggle, *, poll_seconds: float = 0.05):
        self.pause_toggle = pause_toggle
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        if os.name != "nt":
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="tear0-pause-hotkey", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        import ctypes

        get_async_key_state = ctypes.windll.user32.GetAsyncKeyState
        was_pressed = False
        while not self._stop.is_set():
            pressed = self._is_down(get_async_key_state, self.VK_CONTROL) and self._is_down(get_async_key_state, self.VK_OEM_PLUS)
            if pressed and not was_pressed:
                self.pause_toggle.toggle()
            was_pressed = pressed
            time.sleep(self.poll_seconds)

    @staticmethod
    def _is_down(get_async_key_state, virtual_key: int) -> bool:
        return bool(get_async_key_state(virtual_key) & 0x8000)
