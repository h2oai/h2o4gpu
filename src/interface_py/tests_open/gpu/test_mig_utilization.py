# -*- encoding: utf-8 -*-
"""
Task A3 — per-MIG utilization (DCGM branch, Ampere/A100).

Validates that MIG slots carry a real utilization read from the standalone DCGM
host-engine (``DCGM_FI_PROF_GR_ENGINE_ACTIVE`` per GPU-instance).  Requires a
MIG-enabled Ampere box with ``nv-hostengine`` running (datacenter-gpu-manager-4,
core + proprietary).  Skipped automatically elsewhere.

Notes:
  * An *idle* MIG instance legitimately reads 0, so "valid + a reading exists" is the
    primary contract; the ">0 under load" check is gated behind H2O4GPU_MIG_LOAD_TEST=1
    (run it while a workload occupies a slice).
  * ``test_dcgm_reading_per_mig`` is the real A3 gate: it proves the DCGM client
    connected and returned a value for every MIG instance (independent of the value
    being 0 when idle), distinguishing "DCGM works, idle" from "DCGM failed -> 0".
"""
import os
import pytest

from h2o4gpu.util.gpu import (
    get_gpu_slots,
    _mig_instances_by_physical,
    _mig_utilization_dcgm,
)

_SLOTS = []
try:
    _SLOTS = get_gpu_slots(with_usage=True)
except Exception:
    pass
_MIG_SLOTS = [s for s in _SLOTS if s.kind == "mig"]
_HAS_MIG = len(_MIG_SLOTS) > 0

mig_only = pytest.mark.skipif(not _HAS_MIG, reason="No MIG instances present on this host")


@mig_only
def test_mig_utilization_valid():
    """Every MIG slot carries a bounded utilization (0..100)."""
    for s in _MIG_SLOTS:
        assert isinstance(s.utilization, int)
        assert 0 <= s.utilization <= 100, f"{s.cuda_token} util out of range: {s.utilization}"


@mig_only
def test_dcgm_reading_per_mig():
    """DCGM branch returns a live GR_ENGINE_ACTIVE reading for EVERY MIG instance.

    This is the A3 RED->GREEN gate. A non-empty map covering all MIG (physical, gi)
    keys proves the client connected to nv-hostengine, mapped DCGM GPU_I entities to
    NVML GPU-instance ids, and read the metric — even if the values are 0 (idle).
    An empty/partial map means the DCGM path failed (no daemon, missing profiling
    module, struct/version mismatch, or a bad entity mapping).
    """
    mig_by_phys = _mig_instances_by_physical()
    assert mig_by_phys, "expected MIG instances from NVML enumeration"

    util = _mig_utilization_dcgm(mig_by_phys)
    assert util, ("DCGM returned no readings — daemon down, profiling module missing, "
                  "or the ctypes struct/entity mapping needs adjustment")

    expected_keys = {(phys, m["gi"]) for phys, migs in mig_by_phys.items() for m in migs}
    got_keys = set(util)
    assert expected_keys <= got_keys, (
        f"DCGM did not cover every MIG instance.\n  expected {sorted(expected_keys)}\n"
        f"  got      {sorted(got_keys)}\n"
        "  -> likely the DCGM-entity <-> NVML-GI-id mapping (nvmlInstanceId) is off")
    for k in expected_keys:
        assert 0 <= util[k] <= 100, f"util for {k} out of range: {util[k]}"


@pytest.mark.skipif(
    not (_HAS_MIG and os.environ.get("H2O4GPU_MIG_LOAD_TEST") == "1"),
    reason="Set H2O4GPU_MIG_LOAD_TEST=1 and run a GPU load on a MIG slice to check >0",
)
def test_mig_utilization_nonzero_under_load():
    """With a workload occupying a MIG slice, at least one slot reports util > 0."""
    slots = get_gpu_slots(with_usage=True)
    utils = [s.utilization for s in slots if s.kind == "mig"]
    assert max(utils) > 0, f"expected a busy MIG slice to report >0, got {utils}"
