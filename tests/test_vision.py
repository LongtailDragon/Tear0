from tear0.vision import DisplayInfo, _format_display_name, _merge_windows_monitor_names


def test_format_display_name_includes_brand_device_dimensions_and_position():
    display = DisplayInfo(
        index=2,
        left=1680,
        top=-353,
        width=1920,
        height=1080,
        name="",
        device_name="\\\\.\\DISPLAY2",
        monitor_name="Dell U2415",
    )

    assert _format_display_name(display) == "Display 2: Dell U2415 (\\\\.\\DISPLAY2) — 1920x1080 at 1680,-353"


def test_format_display_name_falls_back_when_brand_is_unknown():
    display = DisplayInfo(
        index=1,
        left=0,
        top=0,
        width=1680,
        height=1050,
        name="",
        device_name="\\\\.\\DISPLAY1",
        monitor_name=None,
    )

    assert _format_display_name(display) == "Display 1: \\\\.\\DISPLAY1 — 1680x1050 at 0,0"


def test_merge_windows_monitor_names_matches_by_geometry():
    displays = [
        DisplayInfo(index=1, left=0, top=0, width=1680, height=1050, name="Display 1: old"),
        DisplayInfo(index=2, left=1680, top=-353, width=1920, height=1080, name="Display 2: old"),
    ]
    windows = {
        (1680, -353, 1920, 1080): ("\\\\.\\DISPLAY2", "LG ULTRAWIDE"),
        (0, 0, 1680, 1050): ("\\\\.\\DISPLAY1", "DELL 2007FP"),
    }

    merged = _merge_windows_monitor_names(displays, windows)

    assert merged[0].name == "Display 1: DELL 2007FP (\\\\.\\DISPLAY1) — 1680x1050 at 0,0"
    assert merged[1].name == "Display 2: LG ULTRAWIDE (\\\\.\\DISPLAY2) — 1920x1080 at 1680,-353"
