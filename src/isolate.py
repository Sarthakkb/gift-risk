"""Subprocess isolation for quantum computation.

Qiskit's Rust extensions can crash (SIGSEGV) when executed inside Streamlit's
ScriptRunner threads on macOS. Running the quantum estimators in a fresh
subprocess isolates the native code from the UI process entirely — which is
also the right production shape: risk compute should not live in the
dashboard process.

Implementation note: multiprocessing's spawn start-method re-imports the
parent's main module (app.py under `streamlit run`), which re-executes the
Streamlit script in the child. So instead we launch `python -m src.isolate`
directly and ship the call over stdin/stdout as pickles.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
from importlib import import_module
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_isolated(fn, *args, **kwargs):
    """Execute a picklable module-level function in a fresh subprocess."""
    payload = pickle.dumps((fn.__module__, fn.__qualname__, args, kwargs))
    proc = subprocess.run(
        [sys.executable, "-m", "src.isolate"],
        input=payload,
        capture_output=True,
        cwd=_PROJECT_ROOT,
        timeout=1800,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"isolated worker failed (exit {proc.returncode}): "
            f"{proc.stderr.decode(errors='replace')[-2000:]}"
        )
    status, value = pickle.loads(proc.stdout)
    if status == "err":
        raise RuntimeError(f"isolated worker exception: {value}")
    return value


def _worker_main() -> None:
    module_name, qualname, args, kwargs = pickle.loads(sys.stdin.buffer.read())
    try:
        fn = getattr(import_module(module_name), qualname)
        result = ("ok", fn(*args, **kwargs))
    except Exception as exc:  # ship the error back instead of a stack dump
        result = ("err", f"{type(exc).__name__}: {exc}")
    sys.stdout.buffer.write(pickle.dumps(result))
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    _worker_main()
