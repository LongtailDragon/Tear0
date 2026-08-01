from __future__ import annotations

import ctypes
import json
import platform
import re
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import mss
from PIL import Image

from .output import print_bullet


@dataclass(frozen=True)
class DisplayInfo:
    index: int
    left: int
    top: int
    width: int
    height: int
    name: str
    device_name: str | None = None
    monitor_name: str | None = None


def _format_display_name(display: DisplayInfo) -> str:
    label = display.monitor_name or display.device_name or f"Display {display.index}"
    if display.monitor_name and display.device_name:
        label = f"{display.monitor_name} ({display.device_name})"
    return f"Display {display.index}: {label} — {display.width}x{display.height} at {display.left},{display.top}"


def _merge_windows_monitor_names(
    displays: list[DisplayInfo],
    windows_by_geometry: dict[tuple[int, int, int, int], tuple[str | None, str | None]],
) -> list[DisplayInfo]:
    merged: list[DisplayInfo] = []
    for display in displays:
        key = (display.left, display.top, display.width, display.height)
        device_name, monitor_name = windows_by_geometry.get(key, (display.device_name, display.monitor_name))
        updated = replace(display, device_name=device_name, monitor_name=monitor_name)
        merged.append(replace(updated, name=_format_display_name(updated)))
    return merged


def _decode_wmi_chars(value) -> str | None:
    if not value:
        return None
    chars = []
    for item in value:
        try:
            code = int(item)
        except (TypeError, ValueError):
            continue
        if code:
            chars.append(chr(code))
    text = "".join(chars).strip()
    return text or None


def _windows_wmi_monitor_names() -> dict[str, str]:
    """Return EDID friendly names keyed by PNP/product token, e.g. DELF043."""
    if platform.system() != "Windows":
        return {}
    ps = r"""
$items = Get-CimInstance -Namespace root\wmi -ClassName WmiMonitorID | Where-Object {$_.Active} | ForEach-Object {
  [pscustomobject]@{
    InstanceName = $_.InstanceName
    Name = -join ($_.UserFriendlyName | Where-Object {$_ -ne 0} | ForEach-Object {[char]$_})
  }
}
$items | ConvertTo-Json -Depth 3
"""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            text=True,
            capture_output=True,
            timeout=5,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return {}
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict):
        data = [data]
    names: dict[str, str] = {}
    for item in data:
        instance = str(item.get("InstanceName") or "")
        name = str(item.get("Name") or "").strip()
        if not instance or not name:
            continue
        parts = instance.split("\\")
        if len(parts) >= 2:
            names[parts[1].upper()] = name
    return names


def _windows_monitor_geometry() -> dict[tuple[int, int, int, int], tuple[str | None, str | None]]:
    r"""Map monitor geometry to Win32 display device and EDID friendly name.

    mss gives reliable screenshot coordinates but not monitor names. Win32 gives
    coordinates plus \\.\DISPLAYn, and EnumDisplayDevices gives the monitor PNP
    token. WMI/EDID usually gives the human-friendly brand/model.
    """
    if platform.system() != "Windows":
        return {}

    user32 = ctypes.windll.user32
    wmi_names = _windows_wmi_monitor_names()

    class RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

    class MONITORINFOEXW(ctypes.Structure):
        _fields_ = [
            ("cbSize", ctypes.c_ulong),
            ("rcMonitor", RECT),
            ("rcWork", RECT),
            ("dwFlags", ctypes.c_ulong),
            ("szDevice", ctypes.c_wchar * 32),
        ]

    class DISPLAY_DEVICEW(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("DeviceName", ctypes.c_wchar * 32),
            ("DeviceString", ctypes.c_wchar * 128),
            ("StateFlags", ctypes.c_ulong),
            ("DeviceID", ctypes.c_wchar * 128),
            ("DeviceKey", ctypes.c_wchar * 128),
        ]

    def monitor_name_for_device(device_name: str) -> str | None:
        # EDD_GET_DEVICE_INTERFACE_NAME exposes DISPLAY#DELF043#... so we can
        # join EnumDisplayDevices to WMI EDID friendly names.
        EDD_GET_DEVICE_INTERFACE_NAME = 1
        dd = DISPLAY_DEVICEW()
        dd.cb = ctypes.sizeof(dd)
        if not user32.EnumDisplayDevicesW(device_name, 0, ctypes.byref(dd), EDD_GET_DEVICE_INTERFACE_NAME):
            return None
        match = re.search(r"DISPLAY#([^#]+)#", dd.DeviceID, flags=re.IGNORECASE)
        if match:
            friendly = wmi_names.get(match.group(1).upper())
            if friendly:
                return friendly
        device_string = str(dd.DeviceString or "").strip()
        if device_string and device_string.lower() != "generic pnp monitor":
            return device_string
        return None

    geometry: dict[tuple[int, int, int, int], tuple[str | None, str | None]] = {}
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(RECT), ctypes.c_long)

    def callback(hmonitor, hdc, rect, lparam):
        info = MONITORINFOEXW()
        info.cbSize = ctypes.sizeof(info)
        if user32.GetMonitorInfoW(hmonitor, ctypes.byref(info)):
            device_name = str(info.szDevice)
            rc = info.rcMonitor
            key = (int(rc.left), int(rc.top), int(rc.right - rc.left), int(rc.bottom - rc.top))
            geometry[key] = (device_name, monitor_name_for_device(device_name))
        return 1

    user32.EnumDisplayMonitors(0, 0, callback_type(callback), 0)
    return geometry


def list_displays() -> list[DisplayInfo]:
    with mss.mss() as sct:
        displays = []
        for i, mon in enumerate(sct.monitors[1:], start=1):
            display = DisplayInfo(
                index=i,
                left=int(mon["left"]),
                top=int(mon["top"]),
                width=int(mon["width"]),
                height=int(mon["height"]),
                name="",
            )
            displays.append(replace(display, name=_format_display_name(display)))
        return _merge_windows_monitor_names(displays, _windows_monitor_geometry())


def capture_display(display_index: int, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with mss.mss() as sct:
        if display_index < 1 or display_index >= len(sct.monitors):
            raise ValueError(f"Display {display_index} is not available")
        shot = sct.grab(sct.monitors[display_index])
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        img.save(out_path)
    return out_path


def prompt_for_display() -> DisplayInfo:
    displays = list_displays()
    if not displays:
        raise RuntimeError("No displays were detected")
    print_bullet("Available displays:")
    for d in displays:
        print_bullet(f"{d.index}. {d.name}")
    while True:
        choice = input("> Select display number for Tear0 to view: ").strip()
        try:
            idx = int(choice)
        except ValueError:
            print_bullet("Please enter a display number.")
            continue
        for d in displays:
            if d.index == idx:
                return d
        print_bullet("That display number is not available.")
