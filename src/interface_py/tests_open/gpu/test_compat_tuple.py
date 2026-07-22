# -*- encoding: utf-8 -*-
"""
Task A4 — get_gpu_info_c positional-tuple parity, now slot-based.

The public get_gpu_info_c returns the SAME positional tuple as before, but counts GPU
slots (physical + MIG). On non-MIG hosts it delegates to the physical accessor and is
byte-identical; when MIG is on, arrays are per-slot. Hardware tests skip without a GPU;
the delegation/projection tests are hermetic and run anywhere.
"""
import pytest

import h2o4gpu.util.gpu as g
from h2o4gpu.util.gpu import get_gpu_info_c, get_gpu_slots, num_gpu_slots, GpuSlot

_SLOTS = []
try:
    _SLOTS = get_gpu_slots(with_usage=False)
except Exception:
    pass
gpu_only = pytest.mark.skipif(len(_SLOTS) == 0, reason="No GPU slots available")


# --------------------------- hardware: slot parity ---------------------------

@gpu_only
def test_count_equals_num_slots():
    assert get_gpu_info_c()[0] == num_gpu_slots()


@gpu_only
def test_tuple_positions_and_lengths():
    raw = get_gpu_info_c(return_memory=True, return_name=True, return_usage=True,
                         return_free_memory=True, return_capability=True)
    count = raw[0]
    assert count == num_gpu_slots()
    total_mems, names, usages, free_mems, majors, minors = raw[1:7]
    for arr in (total_mems, names, usages, free_mems, majors, minors):
        assert len(arr) == count


@gpu_only
def test_pid_variant_shapes():
    raw = get_gpu_info_c(return_memory_by_pid=True, return_usage_by_pid=True)
    count = raw[0]
    num_pids, pids, used_mem, num_pids_u, pids_u, used_u = raw[1:7]
    assert len(num_pids) == count and len(num_pids_u) == count
    assert pids.shape[0] == count and pids_u.shape[0] == count


# ------------------ hermetic: delegation + projection logic ------------------

def test_delegates_to_physical_when_no_mig(monkeypatch):
    """No MIG -> byte-identical to the physical accessor (delegation, not re-projection)."""
    monkeypatch.setattr(g, "_mig_instances_by_physical", lambda: {})
    sentinel = ("PHYSICAL_TUPLE",)
    monkeypatch.setattr(g, "_get_gpu_info_c_physical", lambda **kw: sentinel)
    assert g.get_gpu_info_c(return_memory=True, return_usage=True) is sentinel


def _fake_mig_slots(n):
    return [GpuSlot(i, f"MIG-{i}", "mig", False, "1g.5gb", (8, 0),
                    5_000_000_000, 1_000_000_000, 4_000_000_000, i * 10, 0, None)
            for i in range(n)]


def test_slot_based_projection_when_mig(monkeypatch):
    monkeypatch.setattr(g, "_mig_instances_by_physical", lambda: {0: [{"uuid": "MIG-0", "gi": 7}]})
    monkeypatch.setattr(g, "get_gpu_slots", lambda with_usage=True, with_procs=False: _fake_mig_slots(2))
    raw = g.get_gpu_info_c(return_memory=True, return_name=True, return_usage=True,
                           return_free_memory=True, return_capability=True)
    assert raw[0] == 2                                         # count == slot count
    assert list(raw[1]) == [5_000_000_000, 5_000_000_000]      # total_mems
    assert list(raw[2]) == ["1g.5gb", "1g.5gb"]                # names
    assert list(raw[3]) == [0, 10]                             # usages
    assert list(raw[4]) == [4_000_000_000, 4_000_000_000]      # free_mems
    assert list(raw[5]) == [8, 8] and list(raw[6]) == [0, 0]   # majors, minors


def test_slot_based_pid_shapes_when_mig(monkeypatch):
    monkeypatch.setattr(g, "_mig_instances_by_physical", lambda: {0: [{"uuid": "MIG-0", "gi": 7}]})
    monkeypatch.setattr(g, "get_gpu_slots", lambda with_usage=True, with_procs=False: _fake_mig_slots(1))
    raw = g.get_gpu_info_c(return_memory_by_pid=True, return_usage_by_pid=True)
    assert raw[0] == 1
    num_pids, pids, used_mem = raw[1], raw[2], raw[3]
    assert len(num_pids) == 1 and int(num_pids[0]) == 0        # MIG carries no per-pid data
    assert pids.shape == (1, 2000) and used_mem.shape == (1, 2000)
