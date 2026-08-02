"""The backend guard that keeps notebook plots visible.

Regression tests for the failure mode where importing ieeg-backed or cluster
code from a notebook switches matplotlib's backend, so every later plt.show()
draws nothing (with no error).
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import matplotlib
import pytest

from src.analysis.utils.mpl_backend import (
    INLINE_BACKEND, current_backend, ensure_headless_backend, preserve_backend,
    restore_backend,
)


@pytest.fixture
def restore_after():
    """Put the process's backend back after each test."""
    before = current_backend()
    yield
    restore_backend(before)


def hijack_backend():
    """Stand-in for what `ieeg.viz` does at import time on a headless box."""
    try:
        matplotlib.use("Qt5Agg", force=True)
    except Exception:
        print("Qt5Agg backend not available, using default backend")
        matplotlib.use("agg", force=True)


def test_preserve_backend_restores_the_inline_backend(restore_after):
    matplotlib.use(INLINE_BACKEND, force=True)
    with preserve_backend():
        hijack_backend()
    assert current_backend() == INLINE_BACKEND


def test_preserve_backend_restores_after_an_exception(restore_after):
    matplotlib.use(INLINE_BACKEND, force=True)
    with pytest.raises(RuntimeError):
        with preserve_backend():
            hijack_backend()
            raise RuntimeError("import blew up")
    assert current_backend() == INLINE_BACKEND


def test_ensure_headless_backend_respects_a_chosen_backend(restore_after):
    matplotlib.use(INLINE_BACKEND, force=True)
    assert ensure_headless_backend() is False
    assert current_backend() == INLINE_BACKEND


def run_in_fresh_process(snippet):
    """Run `snippet` in a new interpreter (no backend resolved yet) and return stdout."""
    import subprocess
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    code = f"import sys; sys.path.insert(0, {repo_root!r})\n" + snippet
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_ensure_headless_backend_sets_agg_when_nothing_is_chosen():
    """A fresh cluster process still gets Agg -- it never tries to open a window."""
    out = run_in_fresh_process(
        "from src.analysis.utils.mpl_backend import ensure_headless_backend, current_backend\n"
        "print(ensure_headless_backend(), current_backend())\n")
    assert out == "True Agg"


def test_current_backend_does_not_force_resolution():
    """Looking must not resolve the backend, or the guard would defeat itself."""
    out = run_in_fresh_process(
        "from src.analysis.utils.mpl_backend import current_backend\n"
        "print(current_backend())\n")
    assert out == "None"
