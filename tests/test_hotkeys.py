from tear0.hotkeys import PauseToggle, WindowsCtrlPlusHotkey


def test_pause_toggle_switches_state_and_calls_callbacks():
    events = []
    toggle = PauseToggle(on_pause=lambda: events.append("pause"), on_resume=lambda: events.append("resume"))

    assert toggle.toggle() is True
    assert toggle.paused.is_set()
    assert toggle.toggle() is False
    assert not toggle.paused.is_set()
    assert events == ["pause", "resume"]


def test_windows_ctrl_plus_hotkey_uses_control_and_oem_plus():
    pressed = {WindowsCtrlPlusHotkey.VK_CONTROL, WindowsCtrlPlusHotkey.VK_OEM_PLUS}

    def fake_get_async_key_state(virtual_key):
        return 0x8000 if virtual_key in pressed else 0

    assert WindowsCtrlPlusHotkey._is_down(fake_get_async_key_state, WindowsCtrlPlusHotkey.VK_CONTROL)
    assert WindowsCtrlPlusHotkey._is_down(fake_get_async_key_state, WindowsCtrlPlusHotkey.VK_OEM_PLUS)
