"""Spectral distillation pipeline.

Import-order guard for pyarrow.

``pyarrow.dataset`` must be loaded before torch on this platform. torch pulls in
its own OpenMP/MKL DLLs; if ``pyarrow.dataset`` is first imported after that --
which happens implicitly inside ``pandas.read_parquet`` -- its native extension
dies with a Windows access violation (0xC0000005 / exit code -1073741819). The
process dies outright, so there is no traceback and no Python exception to catch:
the Tolokers loader simply never returns.

This module is the earliest hook in the package (it runs before any
``spectral_distillation.src.*`` submodule, and therefore before torch is pulled
in anywhere), which makes loading pyarrow here deterministic and independent of
which entry point runs first.
"""

try:  # pragma: no cover - import-order guard
    import pyarrow.dataset

    PYARROW_DATASET_READY = pyarrow.dataset is not None
except Exception:  # pragma: no cover - pyarrow absent; parquet loaders will say so
    PYARROW_DATASET_READY = False
