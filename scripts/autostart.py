"""Windows autostart configuration for ARA.

Manages ARA as a Windows Task Scheduler task that starts at login.
Launches Ollama and the ARA API server automatically.

Usage:
    uv run python scripts/autostart.py install    # Create startup task
    uv run python scripts/autostart.py uninstall  # Remove startup task
    uv run python scripts/autostart.py status     # Check if task exists
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

TASK_NAME = "ARA-Assistant"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STARTUP_SCRIPT = PROJECT_ROOT / "scripts" / "_ara_startup.bat"

# The batch script that Task Scheduler will execute
STARTUP_BATCH_CONTENT = """@echo off
REM ARA — Automatic startup script
REM Starts Ollama server and ARA API backend

REM Change to project directory
cd /d "{project_root}"

REM Start Ollama in background (if not already running)
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="1" (
    echo Starting Ollama...
    start /B ollama serve
    timeout /T 5 /NOBREAK >NUL
)

REM Start ARA API server
echo Starting ARA API server...
uv run uvicorn ara.api.app:app --host 127.0.0.1 --port 8000
"""

# ── Functions ────────────────────────────────────────────────────────────────


def _run_schtasks(args: list[str], *, capture: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a schtasks command."""
    return subprocess.run(
        ["schtasks", *args],
        capture_output=capture,
        text=True,
    )


def install_task() -> int:
    """Create the Windows Task Scheduler task."""
    # Create the startup batch script
    content = STARTUP_BATCH_CONTENT.format(project_root=PROJECT_ROOT)
    STARTUP_SCRIPT.write_text(content)
    print(f"  Created startup script: {STARTUP_SCRIPT}")

    # Create scheduled task (runs at logon)
    result = _run_schtasks(
        [
            "/Create",
            "/TN",
            TASK_NAME,
            "/TR",
            f'"{STARTUP_SCRIPT}"',
            "/SC",
            "ONLOGON",
            "/RL",
            "HIGHEST",
            "/F",  # Force overwrite if exists
        ]
    )

    if result.returncode != 0:
        print("  FAILED: Could not create scheduled task.")
        print(f"  Error: {result.stderr}")
        print("  TIP: Run this script as Administrator.")
        return 1

    print(f"  Installed task: {TASK_NAME}")
    print("  ARA will start automatically at login.")
    print(f"  To start manually: schtasks /Run /TN {TASK_NAME}")
    return 0


def uninstall_task() -> int:
    """Remove the Windows Task Scheduler task."""
    result = _run_schtasks(["/Delete", "/TN", TASK_NAME, "/F"])

    if result.returncode != 0:
        print(f"  Task '{TASK_NAME}' not found or could not be removed.")
        return 1

    # Clean up the startup script
    if STARTUP_SCRIPT.exists():
        STARTUP_SCRIPT.unlink()
        print(f"  Removed startup script: {STARTUP_SCRIPT}")

    print(f"  Removed task: {TASK_NAME}")
    return 0


def check_status() -> int:
    """Check if the autostart task exists."""
    result = _run_schtasks(["/Query", "/TN", TASK_NAME])

    if result.returncode == 0:
        print(f"  Task '{TASK_NAME}' is installed.")
        # Parse status from output
        for line in result.stdout.split("\n"):
            if "Status" in line or "Next Run" in line or "Last Run" in line:
                print(f"    {line.strip()}")
        return 0

    print(f"  Task '{TASK_NAME}' is NOT installed.")
    return 1


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    """Manage ARA Windows autostart."""
    parser = argparse.ArgumentParser(
        description="ARA Windows autostart configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "action",
        choices=["install", "uninstall", "status"],
        help="Action to perform",
    )
    args = parser.parse_args()

    if args.action == "install":
        return install_task()
    if args.action == "uninstall":
        return uninstall_task()
    return check_status()


if __name__ == "__main__":
    sys.exit(main())
