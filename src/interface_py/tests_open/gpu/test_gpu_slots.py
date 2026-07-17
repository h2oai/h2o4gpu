"""
tests_open/gpu/test_gpu_slots.py
----------------------------------
Task A1 tests for GpuSlot dataclass + get_gpu_slots() physical-GPU projection.

Two test groups:

1.  test_physical_slots_shape  — GPU-BOX ONLY.
    Verifies that get_gpu_slots() returns one GpuSlot per physical GPU with
    the correct field values on a real (non-MIG) GPU host.
    Skip automatically when no GPUs are available (count == 0).

2.  test_projection_*          — PURE-PYTHON, runs on macOS/CPU.
    Patches get_gpu_info_c to return a synthetic 2-device tuple and asserts
    the projection maps every field correctly without needing a GPU or the
    C extension.
"""

import sys
import os
from unittest.mock import patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Resolve the package root so we can import h2o4gpu.util.gpu directly even
# when the package is not installed (i.e. running from the source tree).
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "..", "..")   # …/src/interface_py
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from h2o4gpu.util.gpu import GpuSlot, ProcInfo, get_gpu_slots  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_raw(n: int = 2, with_usage: bool = True, with_procs: bool = False):
    """Return a synthetic get_gpu_info_c positional tuple for n fake devices.

    Mirrors the exact tuple shape produced by get_gpu_info_c when called with:
        return_memory=True, return_name=True, return_usage=with_usage,
        return_free_memory=True, return_capability=True,
        return_memory_by_pid=with_procs, return_usage_by_pid=with_procs,
        return_all=False
    """
    max_pids = 2000

    # Per-device data for two fake GPUs
    total_mems = np.array([8 * 2**30, 16 * 2**30], dtype=np.uint64)[:n]
    free_mems  = np.array([4 * 2**30, 10 * 2**30], dtype=np.uint64)[:n]
    gpu_types  = np.array(["Fake GPU A", "Fake GPU B"])[:n]
    usages     = np.array([42, 77], dtype=np.int32)[:n]
    majors     = np.array([8, 9], dtype=np.int32)[:n]
    minors     = np.array([0, 0], dtype=np.int32)[:n]

    parts = [n, total_mems, gpu_types]
    if with_usage:
        parts.append(usages)
    parts += [free_mems, majors, minors]

    if with_procs:
        num_pids_mem     = np.zeros(n, dtype=np.uint32)
        pids_mem         = np.zeros((n, max_pids), dtype=np.uint32)
        used_gpu_mem     = np.zeros((n, max_pids), dtype=np.uint64)
        num_pids_usage   = np.zeros(n, dtype=np.uint32)
        pids_usage       = np.zeros((n, max_pids), dtype=np.uint32)
        used_gpu_usage   = np.zeros((n, max_pids), dtype=np.uint64)
        parts += [num_pids_mem, pids_mem, used_gpu_mem,
                  num_pids_usage, pids_usage, used_gpu_usage]

    return tuple(parts)


@pytest.fixture(autouse=True)
def _force_physical_projection():
    """A1 covers the physical-GPU projection; force the non-MIG path so these tests are
    hermetic on MIG hosts too (real MIG enumeration/util lives in test_mig_slots.py and
    test_mig_utilization.py). On non-MIG/CPU hosts this patch is a no-op in effect."""
    with patch("h2o4gpu.util.gpu._mig_instances_by_physical", return_value={}):
        yield


# ---------------------------------------------------------------------------
# Group 1 — GPU-box-only integration test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    # Skip on boxes where the C extension is unavailable or returns 0 GPUs.
    # We attempt the import; if it raises (macOS/CPU) or count==0 we skip.
    True,   # Replaced at collection time below via a module-level check.
    reason="No physical GPU available or C extension not importable",
)
def test_physical_slots_shape_placeholder():
    """Placeholder — see the real test below."""


def _gpu_count_safe() -> int:
    """Return GPU count from the C extension, or 0 on import failure."""
    try:
        from h2o4gpu.util.gpu import get_gpu_info_c
        raw = get_gpu_info_c()
        if raw is None:
            return 0
        return raw[0]
    except Exception:
        return 0


_GPU_COUNT = _gpu_count_safe()


@pytest.mark.skipif(_GPU_COUNT == 0, reason="No physical GPU available")
def test_physical_slots_shape():
    """GPU-BOX ONLY: one GpuSlot per physical GPU with expected field values."""
    slots = get_gpu_slots(with_usage=True, with_procs=False)

    # One slot per physical GPU
    assert len(slots) == _GPU_COUNT, (
        f"Expected {_GPU_COUNT} slots, got {len(slots)}"
    )

    # slot_index must be contiguous 0..N-1
    assert [s.slot_index for s in slots] == list(range(len(slots))), (
        "slot_index values are not contiguous 0..N-1"
    )

    for i, slot in enumerate(slots):
        assert isinstance(slot, GpuSlot), f"slot {i} is not a GpuSlot"
        assert slot.kind == "physical", f"slot {i}: kind={slot.kind!r}, expected 'physical'"
        assert slot.groupable is True, f"slot {i}: groupable is not True"
        assert slot.cuda_token == str(i), (
            f"slot {i}: cuda_token={slot.cuda_token!r}, expected {str(i)!r}"
        )
        assert slot.mem_total > 0, f"slot {i}: mem_total={slot.mem_total} must be >0"
        assert 0 <= slot.utilization <= 100, (
            f"slot {i}: utilization={slot.utilization} out of [0,100]"
        )
        assert slot.physical_index == i, (
            f"slot {i}: physical_index={slot.physical_index}, expected {i}"
        )
        assert slot.procs is None, f"slot {i}: procs should be None when with_procs=False"
        assert isinstance(slot.compute_capability, tuple) and len(slot.compute_capability) == 2, (
            f"slot {i}: compute_capability must be a 2-tuple"
        )


# ---------------------------------------------------------------------------
# Group 2 — Pure-Python projection unit tests (no GPU needed)
# ---------------------------------------------------------------------------

def test_projection_count_and_types():
    """Projection: 2 synthetic devices → 2 GpuSlot objects."""
    raw = _make_synthetic_raw(n=2, with_usage=True)
    with patch("h2o4gpu.util.gpu.get_gpu_info_c", return_value=raw):
        slots = get_gpu_slots(with_usage=True)

    assert len(slots) == 2
    for slot in slots:
        assert isinstance(slot, GpuSlot)


def test_projection_slot_index_and_cuda_token():
    """Projection: slot_index and cuda_token are correct for each device."""
    raw = _make_synthetic_raw(n=2, with_usage=True)
    with patch("h2o4gpu.util.gpu.get_gpu_info_c", return_value=raw):
        slots = get_gpu_slots(with_usage=True)

    assert slots[0].slot_index == 0
    assert slots[1].slot_index == 1
    assert slots[0].cuda_token == "0"
    assert slots[1].cuda_token == "1"


def test_projection_kind_and_groupable():
    """Projection: physical devices → kind='physical', groupable=True."""
    raw = _make_synthetic_raw(n=2, with_usage=True)
    with patch("h2o4gpu.util.gpu.get_gpu_info_c", return_value=raw):
        slots = get_gpu_slots(with_usage=True)

    for slot in slots:
        assert slot.kind == "physical"
        assert slot.groupable is True


def test_projection_mem_used_equals_total_minus_free():
    """Projection: mem_used = mem_total - mem_free for each device."""
    raw = _make_synthetic_raw(n=2, with_usage=True)
    # total_mems[0]=8GiB, free_mems[0]=4GiB → used=4GiB
    # total_mems[1]=16GiB, free_mems[1]=10GiB → used=6GiB
    with patch("h2o4gpu.util.gpu.get_gpu_info_c", return_value=raw):
        slots = get_gpu_slots(with_usage=True)

    assert slots[0].mem_total == 8 * 2**30
    assert slots[0].mem_free  == 4 * 2**30
    assert slots[0].mem_used  == 4 * 2**30   # total - free
    assert slots[1].mem_total == 16 * 2**30
    assert slots[1].mem_free  == 10 * 2**30
    assert slots[1].mem_used  == 6 * 2**30


def test_projection_utilization_with_usage_true():
    """Projection: utilization carries through when with_usage=True."""
    raw = _make_synthetic_raw(n=2, with_usage=True)
    # usages = [42, 77]
    with patch("h2o4gpu.util.gpu.get_gpu_info_c", return_value=raw):
        slots = get_gpu_slots(with_usage=True)

    assert slots[0].utilization == 42
    assert slots[1].utilization == 77


def test_projection_utilization_zero_when_usage_false():
    """Projection: utilization is 0 when with_usage=False."""
    raw = _make_synthetic_raw(n=2, with_usage=False)
    with patch("h2o4gpu.util.gpu.get_gpu_info_c", return_value=raw):
        slots = get_gpu_slots(with_usage=False)

    assert slots[0].utilization == 0
    assert slots[1].utilization == 0


def test_projection_compute_capability_tuple():
    """Projection: compute_capability is a (major, minor) tuple per device."""
    raw = _make_synthetic_raw(n=2, with_usage=True)
    # majors=[8, 9], minors=[0, 0]
    with patch("h2o4gpu.util.gpu.get_gpu_info_c", return_value=raw):
        slots = get_gpu_slots(with_usage=True)

    assert slots[0].compute_capability == (8, 0)
    assert slots[1].compute_capability == (9, 0)


def test_projection_name():
    """Projection: name field matches the synthetic gpu_type string."""
    raw = _make_synthetic_raw(n=2, with_usage=True)
    with patch("h2o4gpu.util.gpu.get_gpu_info_c", return_value=raw):
        slots = get_gpu_slots(with_usage=True)

    assert slots[0].name == "Fake GPU A"
    assert slots[1].name == "Fake GPU B"


def test_projection_physical_index():
    """Projection: physical_index equals slot_index for physical GPUs."""
    raw = _make_synthetic_raw(n=2, with_usage=True)
    with patch("h2o4gpu.util.gpu.get_gpu_info_c", return_value=raw):
        slots = get_gpu_slots(with_usage=True)

    assert slots[0].physical_index == 0
    assert slots[1].physical_index == 1


def test_projection_procs_none_when_not_requested():
    """Projection: procs is None when with_procs=False."""
    raw = _make_synthetic_raw(n=2, with_usage=True, with_procs=False)
    with patch("h2o4gpu.util.gpu.get_gpu_info_c", return_value=raw):
        slots = get_gpu_slots(with_usage=True, with_procs=False)

    for slot in slots:
        assert slot.procs is None


def test_projection_returns_empty_on_none():
    """Projection: get_gpu_slots returns [] when get_gpu_info_c returns None."""
    with patch("h2o4gpu.util.gpu.get_gpu_info_c", return_value=None):
        slots = get_gpu_slots()

    assert slots == []


def test_projection_returns_empty_on_zero_count():
    """Projection: get_gpu_slots returns [] when count == 0."""
    raw = (0,)   # count only, no arrays
    with patch("h2o4gpu.util.gpu.get_gpu_info_c", return_value=raw):
        slots = get_gpu_slots()

    assert slots == []


def test_projection_slots_are_frozen():
    """GpuSlot is frozen=True — mutation must raise FrozenInstanceError."""
    raw = _make_synthetic_raw(n=1, with_usage=True)
    with patch("h2o4gpu.util.gpu.get_gpu_info_c", return_value=raw):
        slots = get_gpu_slots(with_usage=True)

    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError (subclass of AttributeError)
        slots[0].utilization = 99  # type: ignore[misc]


def test_projection_contiguous_slot_index():
    """Projection: slot_index values are 0..N-1 for N=2 devices."""
    raw = _make_synthetic_raw(n=2, with_usage=True)
    with patch("h2o4gpu.util.gpu.get_gpu_info_c", return_value=raw):
        slots = get_gpu_slots(with_usage=True)

    assert [s.slot_index for s in slots] == [0, 1]
