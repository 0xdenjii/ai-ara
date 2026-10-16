"""Windows autostart configuration for ARA.

Manages ARA as a Windows Task Scheduler task that starts at login.
Detects ARA_BACKEND_TYPE from .env and launches the appropriate inference
backend (llama-server or Ollama) plus the ARA API server.

Usage:
    uv run python scripts/autostart.py install         # Create startup task
    uv run python scripts/autostart.py uninstall       # Remove startup task
    uv run python scripts/autostart.py status          # Check if task exists
    uv run python scripts/autostart.py launch-backend  # Launch llama-server (used by batch script)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

TASK_NAME = "ARA-Assistant"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STARTUP_SCRIPT = PROJECT_ROOT / "scripts" / "_ara_startup.bat"

# The batch script that Task Scheduler will execute.
# Reads ARA_BACKEND_TYPE from .env to decide which backend to launch.
STARTUP_BATCH_CONTENT = """@echo off
REM ARA — Automatic startup script
REM Detects backend type from .env and starts the appropriate inference server + ARA API.

REM Change to project directory
cd /d "{project_root}"

REM ── Read ARA_BACKEND_TYPE from .env ────────────────────────────────────
set "ARA_BACKEND_TYPE=llamacpp"
for /F "usebackq tokens=1,* delims==" %%A in (".env") do (
    if /I "%%A"=="ARA_BACKEND_TYPE" (
        REM Strip trailing comment and leading/trailing whitespace
        for /F "tokens=1 delims=#" %%C in ("%%B") do (
            for /F "tokens=*" %%D in ("%%C") do set "ARA_BACKEND_TYPE=%%D"
        )
    )
)

echo Detected backend: %ARA_BACKEND_TYPE%

if /I "%ARA_BACKEND_TYPE%"=="ollama" (
    goto :start_ollama
) else (
    goto :start_llamacpp
)

:start_ollama
REM Start Ollama in background (if not already running)
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="1" (
    echo Starting Ollama...
    start /B ollama serve
    timeout /T 5 /NOBREAK >NUL
)
goto :start_ara

:start_llamacpp
REM Start llama-server in background (if not already running)
tasklist /FI "IMAGENAME eq llama-server.exe" 2>NUL | find /I /N "llama-server.exe">NUL
if "%ERRORLEVEL%"=="1" (
    echo Starting llama-server via Python launcher...
    start /B uv run python scripts/autostart.py launch-backend
    timeout /T 10 /NOBREAK >NUL
)
goto :start_ara

:start_ara
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


def _find_model_path() -> str:
    """Locate the primary GGUF model file in the models directory.

    Searches for Q4_K_M GGUFs matching qwen3.5 naming patterns.
    Falls back to the first .gguf file found if no pattern matches.

    Returns:
        Absolute path to the model file.

    Raises:
        FileNotFoundError: If no model files are found.
    """
    models_dir = PROJECT_ROOT / "models"
    if not models_dir.exists():
        msg = f"Models directory not found: {models_dir}. Download a GGUF model first."
        raise FileNotFoundError(msg)

    # Priority: match exact expected filenames first
    priority_patterns = [
        "qwen3.5*35b*Q4_K_M*.gguf",
        "qwen3.5*35b*.gguf",
        "*.Q4_K_M.gguf",
    ]
    for pattern in priority_patterns:
        matches = sorted(models_dir.glob(pattern))
        if matches:
            return str(matches[0])

    # Fallback: any .gguf file
    all_gguf = sorted(models_dir.glob("*.gguf"))
    if all_gguf:
        return str(all_gguf[0])

    msg = f"No .gguf model files found in {models_dir}. Download a model first."
    raise FileNotFoundError(msg)


def launch_backend() -> int:
    """Launch the llama-server process with args derived from ARA settings.

    Called by the batch startup script when ARA_BACKEND_TYPE=llamacpp.
    Reads all settings from .env / settings.py, locates the model file,
    builds the correct CLI args, and exec's llama-server.
    """
    # Import settings here to avoid import cost for install/uninstall/status
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from ara.config.settings import AraSettings  # type: ignore[import-untyped]
    from ara.core.llamacpp import build_server_args  # type: ignore[import-untyped]

    settings = AraSettings()

    if settings.llm.backend_type != "llamacpp":
        print(
            f"  Backend type is '{settings.llm.backend_type}', not 'llamacpp'. Skipping llama-server launch."
        )
        return 0

    # Find llama-server executable
    llama_server_exe = shutil.which("llama-server")
    if llama_server_exe is None:
        # Check common locations
        for candidate in [
            PROJECT_ROOT / "llama-server.exe",
            Path(os.environ.get("LLAMACPP_DIR", "")) / "llama-server.exe",
            Path.home() / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe",
        ]:
            if candidate.exists():
                llama_server_exe = str(candidate)
                break

    if llama_server_exe is None:
        print("  ERROR: llama-server not found on PATH or in common locations.")
        print("  Install llama.cpp and ensure llama-server is accessible.")
        print("  Hint: set LLAMACPP_DIR env var or add llama-server to PATH.")
        return 1

    try:
        model_path = _find_model_path()
    except FileNotFoundError as exc:
        print(f"  ERROR: {exc}")
        return 1

    print(f"  Model: {model_path}")
    print(f"  Server: {llama_server_exe}")

    # Build args from settings
    server_args = build_server_args(settings, model_path)

    # Add --jinja for template processing and --device for GPU selection
    extra_flags: list[str] = []
    if "--jinja" not in server_args:
        extra_flags.append("--jinja")
    # --device CUDA0 for explicit GPU selection (recommended for single-GPU)
    if "--device" not in server_args:
        extra_flags.extend(["--device", "CUDA0"])
    # --fit on for automatic context fitting
    if "--fit" not in server_args:
        extra_flags.extend(["--fit", "on"])

    full_args = [llama_server_exe, *server_args, *extra_flags]
    print(f"  Launching: {' '.join(full_args)}")

    # Replace this process with llama-server
    os.execv(llama_server_exe, full_args)
    return 0  # unreachable but satisfies type checker


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    """Manage ARA Windows autostart."""
    parser = argparse.ArgumentParser(
        description="ARA Windows autostart configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "action",
        choices=["install", "uninstall", "status", "launch-backend"],
        help="Action to perform",
    )
    args = parser.parse_args()

    if args.action == "install":
        return install_task()
    if args.action == "uninstall":
        return uninstall_task()
    if args.action == "launch-backend":
        return launch_backend()
    return check_status()


if __name__ == "__main__":
    sys.exit(main())
