# -*- encoding: utf-8 -*-
"""MIG-only container: the physical NVML enumeration bails (parent
nvmlDeviceGetMemoryInfo -> NOT_SUPPORTED -> C returns None), but the per-MIG-device
NVML path still works. get_gpu_slots must fall back to building slots straight from the
MIG instances instead of returning []. Hermetic (mocks NVML) — no GPU needed."""
import numpy as np
import h2o4gpu.util.gpu as g
from h2o4gpu.util.gpu import get_gpu_slots


def _mig4(with_procs=False, busy=False):
    def inst(uuid, gi, procs):
        return {"uuid": uuid, "name": "1g.5gb", "mem_total": 5 * 2**30,
                "mem_free": 4 * 2**30, "cc": (8, 0), "gi": gi, "ci": 0,
                "procs": procs if with_procs else None}
    return {0: [inst("MIG-aaaa", 7, [(1234, 1073741824)] if busy else []),
                inst("MIG-bbbb", 8, []),
                inst("MIG-cccc", 9, []),
                inst("MIG-dddd", 11, [])]}


def test_mig_only_container_physical_none(monkeypatch):
    # parent memory NOT_SUPPORTED -> C enumeration returns None
    monkeypatch.setattr(g, "_get_gpu_info_c_physical", lambda **kw: None)
    monkeypatch.setattr(g, "_mig_instances_by_physical", lambda with_procs=False: _mig4())
    slots = get_gpu_slots(with_usage=False)
    assert len(slots) == 4
    assert all(s.kind == "mig" for s in slots)
    assert [s.cuda_token for s in slots] == ["MIG-aaaa", "MIG-bbbb", "MIG-cccc", "MIG-dddd"]
    assert [s.slot_index for s in slots] == [0, 1, 2, 3]      # reindexed 0..N-1
    assert all(not s.groupable and s.physical_index == 0 for s in slots)


def test_mig_only_container_physical_zero_count(monkeypatch):
    zero = (0, np.array([], np.uint64), np.array([]), np.array([], np.int32),
            np.array([], np.uint64), np.array([], np.int32), np.array([], np.int32))
    monkeypatch.setattr(g, "_get_gpu_info_c_physical", lambda **kw: zero)
    monkeypatch.setattr(g, "_mig_instances_by_physical", lambda with_procs=False: _mig4())
    slots = get_gpu_slots(with_usage=False)
    assert len(slots) == 4 and all(s.kind == "mig" for s in slots)


def test_no_physical_no_mig_returns_empty(monkeypatch):
    # non-MIG container with no visible GPU: nothing to enumerate
    monkeypatch.setattr(g, "_get_gpu_info_c_physical", lambda **kw: None)
    monkeypatch.setattr(g, "_mig_instances_by_physical", lambda with_procs=False: {})
    assert get_gpu_slots(with_usage=False) == []


def test_mig_only_container_carries_procs(monkeypatch):
    monkeypatch.setattr(g, "_get_gpu_info_c_physical", lambda **kw: None)
    monkeypatch.setattr(g, "_mig_instances_by_physical",
                        lambda with_procs=False: _mig4(with_procs=with_procs, busy=True))
    slots = get_gpu_slots(with_usage=False, with_procs=True)
    busy = [s for s in slots if s.procs]
    assert len(busy) == 1
    assert busy[0].cuda_token == "MIG-aaaa"
    assert busy[0].procs[0].pid == 1234 and busy[0].procs[0].used_mem == 1073741824
