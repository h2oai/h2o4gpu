# -*- encoding: utf-8 -*-
"""
Task A7 — public import surface + no DCGM/NVML leak + bundled lib shipped.

Runs anywhere (no GPU needed): these are import/packaging invariants, not hardware tests.
"""
import importlib
import os

import pytest

_PUBLIC = [
    "get_gpu_info_c", "get_gpu_slots", "num_gpu_slots",
    "cuda_token", "cuda_tokens", "cuda_vis_check", "gpu_utilization_watch",
]


def test_public_surface_imports():
    mod = importlib.import_module("h2o4gpu.util.gpu")
    for name in _PUBLIC:
        assert hasattr(mod, name) and callable(getattr(mod, name)), \
            f"public symbol missing/not callable: {name}"


def test_public_surface_direct_import():
    # the exact form DAI / consumers use
    from h2o4gpu.util.gpu import (  # noqa: F401
        get_gpu_info_c, get_gpu_slots, num_gpu_slots,
        cuda_token, cuda_tokens, cuda_vis_check, gpu_utilization_watch,
    )


def test_no_dcgm_or_nvml_symbols_leak_public():
    """Importing the module must not expose DCGM/GPM/NVML symbols in the public namespace.

    The DCGM client is fully private (underscore-prefixed classes/helpers) and the shared
    lib is only dlopen'd lazily inside _DcgmMigSession.open() — never at import.
    """
    mod = importlib.import_module("h2o4gpu.util.gpu")
    leaked = [n for n in dir(mod)
              if not n.startswith("_")
              and any(tok in n.lower() for tok in ("dcgm", "nvml", "gpm"))]
    assert not leaked, f"DCGM/NVML/GPM symbols leaked into the public API: {leaked}"


def test_import_does_not_dlopen_dcgm():
    """Importing the module must not dlopen libdcgm — the shared lib is loaded lazily
    only inside _DcgmMigSession.open(). Verified in a fresh interpreter via /proc maps."""
    if not os.path.exists("/proc/self/maps"):
        pytest.skip("needs /proc (Linux only)")
    import subprocess
    import sys
    code = ("import h2o4gpu.util.gpu; "
            "print(any('libdcgm' in line for line in open('/proc/self/maps')))")
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out == "False", "importing h2o4gpu.util.gpu eagerly dlopen'd libdcgm"


def test_bundled_libdcgm_shipped():
    """The Apache-2.0 core libdcgm.so.4.5.3 ships inside the package (not a pip dep)."""
    import h2o4gpu.util.gpu as g
    pkg_root = os.path.dirname(os.path.dirname(g.__file__))   # …/h2o4gpu
    lib = os.path.join(pkg_root, "lib", "libdcgm.so.4.5.3")
    assert os.path.isfile(lib), f"bundled libdcgm.so.4.5.3 not found at {lib}"
