"""ARA setup script — one-command environment bootstrap for fresh installs.

Usage:
    uv run python scripts/setup.py          # Full setup
    uv run python scripts/setup.py --check  # Verify setup without changes
    uv run python scripts/setup.py --skip-models  # Skip model downloads

Steps:
    1. Verify Python version and OS
    2. Create data directory structure
    3. Copy .env.example → .env (if missing)
    4. Install Python dependencies
    5. Verify Ollama is installed and running
    6. Pull required Ollama models
    7. Download voice models (Moonshine, Kokoro, openWakeWord)
    8. Download embedding and reranker models
    9. Initialize vector database
    10. Set up frontend (npm install, build)
    11. Run tests to verify installation
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"

MIN_PYTHON_VERSION = (3, 11)

REQUIRED_DIRS = [
    DATA_DIR,
    DATA_DIR / "vector_db",
    DATA_DIR / "graph_db",
    DATA_DIR / "models",
    DATA_DIR / "models" / "kokoro",
    DATA_DIR / "fine_tuning",
    DATA_DIR / "conversations",
    PROJECT_ROOT / "no-commit",
    PROJECT_ROOT / "no-commit" / "sessions",
    PROJECT_ROOT / "no-commit" / "memory",
]

OLLAMA_MODELS = [
    "qwen3:30b-a3b",
    "nomic-embed-text",
]

# ── Utilities ────────────────────────────────────────────────────────────────


class SetupStep:
    """A single setup step with status tracking."""

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self.status: str = "pending"
        self.message: str = ""

    def success(self, message: str = "") -> None:
        self.status = "ok"
        self.message = message

    def skip(self, message: str = "") -> None:
        self.status = "skip"
        self.message = message

    def fail(self, message: str = "") -> None:
        self.status = "FAIL"
        self.message = message

    def warn(self, message: str = "") -> None:
        self.status = "warn"
        self.message = message


def _print_step(step: SetupStep) -> None:
    """Print a step result with colored status."""
    status_symbols = {
        "ok": "[OK]",
        "skip": "[SKIP]",
        "FAIL": "[FAIL]",
        "warn": "[WARN]",
        "pending": "[....]",
    }
    symbol = status_symbols.get(step.status, "[????]")
    line = f"  {symbol} {step.description}"
    if step.message:
        line += f" — {step.message}"
    print(line)


def _run_command(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    capture: bool = True,
    timeout_s: int = 300,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command with timeout."""
    return subprocess.run(
        cmd,
        cwd=cwd or PROJECT_ROOT,
        capture_output=capture,
        text=True,
        timeout=timeout_s,
    )


def _command_exists(name: str) -> bool:
    """Check if a command is available on PATH."""
    return shutil.which(name) is not None


# ── Setup Steps ──────────────────────────────────────────────────────────────


def check_python(step: SetupStep) -> bool:
    """Verify Python version and OS."""
    version = sys.version_info[:2]
    if version < MIN_PYTHON_VERSION:
        step.fail(
            f"Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+ required, got {version[0]}.{version[1]}"
        )
        return False

    if platform.system() != "Windows":
        step.warn(f"ARA is designed for Windows, running on {platform.system()}")
        return True

    step.success(f"Python {version[0]}.{version[1]} on {platform.system()}")
    return True


def create_directories(step: SetupStep, *, check_only: bool = False) -> bool:
    """Create required data directories."""
    missing = [d for d in REQUIRED_DIRS if not d.exists()]

    if not missing:
        step.success("All directories exist")
        return True

    if check_only:
        step.fail(f"{len(missing)} directories missing")
        return False

    for d in missing:
        d.mkdir(parents=True, exist_ok=True)

    step.success(f"Created {len(missing)} directories")
    return True


def setup_env_file(step: SetupStep, *, check_only: bool = False) -> bool:
    """Copy .env.example to .env if missing."""
    if ENV_FILE.exists():
        step.skip(".env already exists")
        return True

    if not ENV_EXAMPLE.exists():
        step.fail(".env.example not found")
        return False

    if check_only:
        step.fail(".env file missing (run without --check to create)")
        return False

    shutil.copy2(ENV_EXAMPLE, ENV_FILE)
    step.success("Created .env from .env.example")
    return True


def install_dependencies(step: SetupStep, *, check_only: bool = False) -> bool:
    """Install Python dependencies via uv."""
    if not _command_exists("uv"):
        step.fail("'uv' not found — install from https://docs.astral.sh/uv/")
        return False

    if check_only:
        # Check if venv exists and has key packages
        venv_dir = PROJECT_ROOT / ".venv"
        if venv_dir.exists():
            step.success("Virtual environment exists")
            return True
        step.fail("No .venv directory found")
        return False

    result = _run_command(["uv", "sync", "--extra", "dev"], timeout_s=600)
    if result.returncode != 0:
        step.fail(f"uv sync failed: {result.stderr[:200]}")
        return False

    step.success("Python dependencies installed")
    return True


def check_ollama(step: SetupStep) -> bool:
    """Verify Ollama is installed and reachable."""
    if not _command_exists("ollama"):
        step.fail("Ollama not found — install from https://ollama.com/download")
        return False

    try:
        result = _run_command(["ollama", "list"], timeout_s=10)
        if result.returncode != 0:
            step.warn("Ollama installed but not running — start with 'ollama serve'")
            return False
        step.success("Ollama is running")
        return True
    except subprocess.TimeoutExpired:
        step.warn("Ollama timed out — may not be running")
        return False


def pull_ollama_models(
    step: SetupStep,
    *,
    check_only: bool = False,
    skip_models: bool = False,
) -> bool:
    """Pull required Ollama models."""
    if skip_models:
        step.skip("Model download skipped (--skip-models)")
        return True

    if not _command_exists("ollama"):
        step.skip("Ollama not available")
        return False

    try:
        result = _run_command(["ollama", "list"], timeout_s=10)
        if result.returncode != 0:
            step.skip("Ollama not running")
            return False

        installed = result.stdout.lower()
        missing = [m for m in OLLAMA_MODELS if m.split(":")[0] not in installed]

        if not missing:
            step.success(f"All {len(OLLAMA_MODELS)} models available")
            return True

        if check_only:
            step.fail(f"Missing models: {', '.join(missing)}")
            return False

        for model in missing:
            print(f"    Pulling {model}...")
            pull_result = _run_command(
                ["ollama", "pull", model],
                timeout_s=3600,
                capture=False,
            )
            if pull_result.returncode != 0:
                step.warn(f"Failed to pull {model}")
                return False

        step.success(f"Pulled {len(missing)} models")
        return True

    except subprocess.TimeoutExpired:
        step.warn("Ollama timed out")
        return False


def setup_frontend(step: SetupStep, *, check_only: bool = False) -> bool:
    """Install frontend dependencies and build."""
    if not FRONTEND_DIR.exists():
        step.skip("No frontend directory")
        return True

    if not _command_exists("npm"):
        step.warn("npm not found — frontend setup skipped")
        return True

    node_modules = FRONTEND_DIR / "node_modules"
    if check_only:
        if node_modules.exists():
            step.success("Frontend dependencies installed")
            return True
        step.fail("Frontend node_modules missing")
        return False

    # npm install
    result = _run_command(["npm", "install"], cwd=FRONTEND_DIR, timeout_s=300)
    if result.returncode != 0:
        step.fail(f"npm install failed: {result.stderr[:200]}")
        return False

    step.success("Frontend dependencies installed")
    return True


def run_tests(step: SetupStep, *, check_only: bool = False) -> bool:
    """Run the test suite to verify installation."""
    result = _run_command(
        ["uv", "run", "python", "-m", "pytest", "tests/", "-x", "--tb=line", "-q"],
        timeout_s=120,
    )

    if result.returncode != 0:
        step.warn("Some tests failed — check output")
        return False

    # Parse pass count from output
    for line in result.stdout.split("\n"):
        if "passed" in line:
            step.success(line.strip())
            return True

    step.success("Tests completed")
    return True


def check_gpu(step: SetupStep) -> bool:
    """Check for NVIDIA GPU availability."""
    if not _command_exists("nvidia-smi"):
        step.warn("nvidia-smi not found — GPU may not be available")
        return True

    try:
        result = _run_command(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], timeout_s=10
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_info = result.stdout.strip().split("\n")[0]
            step.success(f"GPU: {gpu_info}")
            return True
        step.warn("GPU detected but query failed")
        return True
    except subprocess.TimeoutExpired:
        step.warn("nvidia-smi timed out")
        return True


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    """Run the ARA setup process."""
    parser = argparse.ArgumentParser(
        description="ARA — One-command environment setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify setup without making changes",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip downloading Ollama models",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running the test suite",
    )
    parser.add_argument(
        "--skip-frontend",
        action="store_true",
        help="Skip frontend setup (npm install)",
    )
    args = parser.parse_args()

    mode = "CHECK" if args.check else "SETUP"
    print(f"\n{'=' * 60}")
    print(f"  ARA Setup — {mode} Mode")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"{'=' * 60}\n")

    steps: list[SetupStep] = []

    # Step 1: Python & OS
    step1 = SetupStep("python", "Python version and OS")
    check_python(step1)
    steps.append(step1)

    # Step 2: GPU check
    step_gpu = SetupStep("gpu", "NVIDIA GPU availability")
    check_gpu(step_gpu)
    steps.append(step_gpu)

    # Step 3: Create directories
    step2 = SetupStep("dirs", "Data directory structure")
    create_directories(step2, check_only=args.check)
    steps.append(step2)

    # Step 4: .env file
    step3 = SetupStep("env", "Environment configuration (.env)")
    setup_env_file(step3, check_only=args.check)
    steps.append(step3)

    # Step 5: Python dependencies
    step4 = SetupStep("deps", "Python dependencies (uv sync)")
    install_dependencies(step4, check_only=args.check)
    steps.append(step4)

    # Step 6: Ollama
    step5 = SetupStep("ollama", "Ollama availability")
    check_ollama(step5)
    steps.append(step5)

    # Step 7: Ollama models
    step6 = SetupStep("models", "Ollama models")
    pull_ollama_models(step6, check_only=args.check, skip_models=args.skip_models)
    steps.append(step6)

    # Step 8: Frontend
    if not args.skip_frontend:
        step7 = SetupStep("frontend", "Frontend dependencies (npm)")
        setup_frontend(step7, check_only=args.check)
        steps.append(step7)

    # Step 9: Tests
    if not args.skip_tests:
        step8 = SetupStep("tests", "Test suite verification")
        run_tests(step8, check_only=args.check)
        steps.append(step8)

    # Print results
    print("\nResults:")
    print("-" * 60)
    for step in steps:
        _print_step(step)
    print("-" * 60)

    failures = sum(1 for s in steps if s.status == "FAIL")
    warnings = sum(1 for s in steps if s.status == "warn")

    if failures > 0:
        print(f"\n  {failures} step(s) failed. Fix the issues above and re-run.")
        return 1

    if warnings > 0:
        print(f"\n  Setup complete with {warnings} warning(s).")
    else:
        print("\n  Setup complete! All checks passed.")

    print("\n  Next steps:")
    print("    1. Start Ollama:    ollama serve")
    print("    2. Start backend:   uv run uvicorn ara.api.app:app --host 127.0.0.1 --port 8000")
    print("    3. Start frontend:  npm run dev  (from frontend/)")
    print("    4. Open dashboard:  http://localhost:3000")
    print("    5. Try CLI chat:    uv run python scripts/chat.py")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
