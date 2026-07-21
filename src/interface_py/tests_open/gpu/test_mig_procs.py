# -*- encoding: utf-8 -*-
"""MIG per-process memory: projection from _mig_instances_by_physical procs into
GpuSlot.procs and the get_gpu_info_c(return_memory_by_pid=True) tuple. Hermetic
(mocks the NVML enumeration) — runs anywhere, no GPU needed."""
from unittest.mock import patch
import h2o4gpu.util.gpu as g
from h2o4gpu.util.gpu import get_gpu_slots, get_gpu_info_c, GpuSlot, ProcInfo


def _one_physical(with_usage=True, **kw):
    # minimal physical tuple for a 1-GPU parent: count + memory/name/usage/free/major/minor
    import numpy as np
    return (1,
            np.array([40 * 2**30], dtype=np.uint64),   # total_mems
            np.array(["A100"]),                         # names
            np.array([0], dtype=np.int32),              # usages
            np.array([40 * 2**30], dtype=np.uint64),    # free_mems
            np.array([8], dtype=np.int32),              # majors
            np.array([0], dtype=np.int32))              # minors


_MIG = {0: [{"uuid": "MIG-aaaa", "name": "1g.5gb", "mem_total": 5 * 2**30,
             "mem_free": 4 * 2**30, "cc": (8, 0), "gi": 7, "ci": 0,
             "procs": [(1234, 1073741824)]}]}   # 1234 -> 1024 MiB


def test_mig_slot_carries_procinfo(monkeypatch):
    monkeypatch.setattr(g, "_get_gpu_info_c_physical", _one_physical)
    monkeypatch.setattr(g, "_mig_instances_by_physical",
                        lambda with_procs=False: _MIG if with_procs else
                        {0: [{**_MIG[0][0], "procs": None}]})
    slots = get_gpu_slots(with_usage=False, with_procs=True)
    mig = [s for s in slots if s.kind == "mig"]
    assert len(mig) == 1
    assert mig[0].procs == [ProcInfo(pid=1234, used_mem=1073741824, usage=0)]


def test_get_gpu_info_c_reports_mig_pid_memory(monkeypatch):
    monkeypatch.setattr(g, "_get_gpu_info_c_physical", _one_physical)
    monkeypatch.setattr(g, "_mig_instances_by_physical",
                        lambda with_procs=False: _MIG if with_procs else
                        {0: [{**_MIG[0][0], "procs": None}]})
    raw = get_gpu_info_c(return_memory_by_pid=True)
    count = raw[0]
    num_pids, pids, used = raw[1], raw[2], raw[3]
    idx = 0  # single MIG slot
    assert int(num_pids[idx]) == 1
    assert int(pids[idx, 0]) == 1234
    assert int(used[idx, 0]) == 1073741824


def test_with_procs_false_leaves_mig_procs_none(monkeypatch):
    monkeypatch.setattr(g, "_get_gpu_info_c_physical", _one_physical)
    monkeypatch.setattr(g, "_mig_instances_by_physical",
                        lambda with_procs=False: {0: [{**_MIG[0][0], "procs": None}]})
    slots = get_gpu_slots(with_usage=False, with_procs=False)
    mig = [s for s in slots if s.kind == "mig"]
    assert mig and mig[0].procs is None
