"""
tests_open/gpu/test_dcgm_bundled.py
------------------------------------
Tests for the DCGM 4.5.3 bundle layout: bundled .so, C headers, and absence of
the superseded Python bindings.

All tests are pure-Python and run on macOS/CPU boxes.
"""

import os
import importlib.util

import pytest

# ---------------------------------------------------------------------------
# Repo-root resolution: walk up from this file's location
# ---------------------------------------------------------------------------

def _repo_root() -> str:
    """Return the absolute path to the h2o4gpu repo root (contains CMakeLists.txt)."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = here
    for _ in range(10):
        if os.path.isfile(os.path.join(candidate, "CMakeLists.txt")):
            return candidate
        candidate = os.path.dirname(candidate)
    raise RuntimeError(f"Could not find repo root from {here!r}")


_REPO_ROOT = _repo_root()
_LIB_DIR = os.path.join(_REPO_ROOT, "src", "interface_py", "h2o4gpu", "lib")
_UTIL_DIR = os.path.join(_REPO_ROOT, "src", "interface_py", "h2o4gpu", "util")
_THIRD_PARTY_DCGM = os.path.join(_REPO_ROOT, "third_party", "dcgm")


# ---------------------------------------------------------------------------
# 1. Bundled .so is libdcgm.so.4.5.3 (present, >1 MB); 4.6.0 is gone
# ---------------------------------------------------------------------------

def test_bundled_so_453_present():
    """libdcgm.so.4.5.3 must be present in h2o4gpu/lib/ and be >1 MB."""
    path = os.path.join(_LIB_DIR, "libdcgm.so.4.5.3")
    assert os.path.isfile(path), f"libdcgm.so.4.5.3 not found at: {path!r}"
    size = os.path.getsize(path)
    assert size > 1 * 1024 * 1024, (
        f"libdcgm.so.4.5.3 is suspiciously small ({size} bytes); expected >1 MB"
    )


def test_bundled_so_460_absent():
    """libdcgm.so.4.6.0 must NOT be present in h2o4gpu/lib/ (superseded by 4.5.3)."""
    stale = os.path.join(_LIB_DIR, "libdcgm.so.4.6.0")
    assert not os.path.exists(stale), (
        f"Stale libdcgm.so.4.6.0 found at {stale!r} — must be deleted"
    )


# ---------------------------------------------------------------------------
# 2. Python bindings (dcgm_bindings/ and _dcgm_loader.py) no longer exist
# ---------------------------------------------------------------------------

def test_dcgm_bindings_dir_absent():
    """dcgm_bindings/ directory must not exist (DCGM client is now C++ in dcgm_mig.cpp)."""
    bindings_dir = os.path.join(_UTIL_DIR, "dcgm_bindings")
    assert not os.path.exists(bindings_dir), (
        f"dcgm_bindings/ still present at {bindings_dir!r} — must be deleted"
    )


def test_dcgm_loader_absent():
    """_dcgm_loader.py must not exist (Python DCGM loader superseded by C++ client)."""
    loader = os.path.join(_UTIL_DIR, "_dcgm_loader.py")
    assert not os.path.exists(loader), (
        f"_dcgm_loader.py still present at {loader!r} — must be deleted"
    )


# ---------------------------------------------------------------------------
# 3. third_party/dcgm/ has the 4 C headers + LICENSE
# ---------------------------------------------------------------------------

_REQUIRED_HEADERS = [
    "dcgm_agent.h",
    "dcgm_structs.h",
    "dcgm_fields.h",
    "dcgm_api_export.h",
    "LICENSE",
]


@pytest.mark.parametrize("filename", _REQUIRED_HEADERS)
def test_third_party_dcgm_file_present(filename):
    """Each of the 4 vendored C headers and LICENSE must exist in third_party/dcgm/."""
    path = os.path.join(_THIRD_PARTY_DCGM, filename)
    assert os.path.isfile(path), (
        f"Expected vendor file not found: {path!r}"
    )
    assert os.path.getsize(path) > 0, f"File is empty: {path!r}"


def test_third_party_dcgm_headers_have_copyright():
    """All 4 C headers must contain 'Copyright' and 'NVIDIA'."""
    headers = [f for f in _REQUIRED_HEADERS if f.endswith(".h")]
    for name in headers:
        path = os.path.join(_THIRD_PARTY_DCGM, name)
        content = open(path, encoding="utf-8").read()
        assert "Copyright" in content, f"{name}: missing 'Copyright'"
        assert "NVIDIA" in content, f"{name}: missing 'NVIDIA'"


def test_third_party_dcgm_license_is_apache():
    """The vendored LICENSE must be Apache 2.0."""
    path = os.path.join(_THIRD_PARTY_DCGM, "LICENSE")
    content = open(path, encoding="utf-8").read()
    assert "Apache License" in content, "LICENSE does not contain 'Apache License'"
    assert "Version 2.0" in content, "LICENSE does not contain 'Version 2.0'"
