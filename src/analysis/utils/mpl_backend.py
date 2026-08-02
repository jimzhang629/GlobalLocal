"""Keep matplotlib's backend from being hijacked by library imports.

``ieeg.viz.*`` calls ``matplotlib.use("Qt5Agg")`` at import time. On a headless
machine -- the DCC, or any Jupyter kernel without an X server -- Qt isn't
available, so matplotlib prints

    Qt5Agg backend not available, using default backend

and leaves the process on the non-interactive default (``Agg``). That message is
harmless on its own, but the backend switch is not: in a notebook that had
already run ``%matplotlib inline``, every later ``plt.show()`` silently draws
nothing.

Wrap imports that might switch the backend in :func:`preserve_backend`, and call
:func:`use_inline_backend` from a notebook to put the inline backend back after
the fact.
"""

from contextlib import contextmanager

import matplotlib

INLINE_BACKEND = "module://matplotlib_inline.backend_inline"


def current_backend():
    """The selected backend, or None if matplotlib hasn't resolved one yet.

    Read through ``dict.__getitem__`` so that merely looking doesn't trigger
    matplotlib's lazy resolution -- ``rcParams["backend"]`` picks and imports a
    backend when it still holds the auto sentinel.
    """
    backend = dict.__getitem__(matplotlib.rcParams, "backend")
    return backend if isinstance(backend, str) else None


def restore_backend(backend):
    """Switch back to `backend`. No-op if it's None or already active."""
    if backend is None or current_backend() == backend:
        return False
    try:
        matplotlib.use(backend, force=True)
    except Exception:  # the previous backend is genuinely unusable here
        return False
    return True


@contextmanager
def preserve_backend():
    """Undo any backend switch performed inside the block."""
    before = current_backend()
    try:
        yield
    finally:
        restore_backend(before)


def ensure_headless_backend(backend="Agg"):
    """Select a non-interactive backend, unless a backend is already chosen.

    Cluster scripts want ``matplotlib.use("Agg")`` so they never try to open a
    window. Calling that unconditionally at import time is a trap, though: these
    scripts are also imported *from notebooks* for their helpers, and the switch
    kills ``%matplotlib inline`` -- plots stop appearing with no error at all.

    A fresh script process hasn't resolved a backend yet, so it gets `backend`;
    a notebook already picked one with ``%matplotlib inline``, so it is left
    alone. Returns True if the backend was switched.
    """
    if current_backend() is not None:
        return False
    matplotlib.use(backend, force=True)
    return True


def use_inline_backend(verbose=True):
    """Put the Jupyter inline backend back, so ``plt.show()`` draws again.

    Call this after importing anything that pulls in ``ieeg.viz`` if figures
    have stopped appearing. Outside IPython (or without matplotlib_inline
    installed) the backend is left alone and this returns False.
    """
    if current_backend() == INLINE_BACKEND:
        return True

    ipython = None
    try:
        from IPython import get_ipython
        ipython = get_ipython()
    except Exception:
        pass

    try:
        if ipython is not None:
            # Preferred: same path as `%matplotlib inline`, so IPython also
            # re-registers its post-execute figure flush.
            ipython.run_line_magic("matplotlib", "inline")
        else:
            matplotlib.use(INLINE_BACKEND, force=True)
    except Exception as e:
        if verbose:
            print(f"could not select the inline backend ({type(e).__name__}: {e})"
                  f" -- still on {current_backend()!r}")
        return False

    if verbose:
        print("matplotlib backend ->", current_backend())
    return True
