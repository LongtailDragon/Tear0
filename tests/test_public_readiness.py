import subprocess
import sys


def test_public_readiness_script_passes_on_tracked_files():
    result = subprocess.run(
        [sys.executable, "scripts/public_readiness_check.py"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
