# -*- encoding: utf-8 -*-
"""
Task A6 — gpu_utilization_watch(): persistent, low-overhead utilization sampler.

Runs on any GPU host (physical or MIG); skipped where no GPU is visible. MIG and
physical slots look identical to the caller ({slot_index: util%}). Idle utilization
of 0 is valid; the ">0 under load" check is gated behind H2O4GPU_MIG_LOAD_TEST=1.
"""
import os
import pytest

from h2o4gpu.util.gpu import get_gpu_slots, gpu_utilization_watch

_SLOTS = []
try:
    _SLOTS = get_gpu_slots(with_usage=False)
except Exception:
    pass
gpu_only = pytest.mark.skipif(len(_SLOTS) == 0, reason="No GPU slots available")


@gpu_only
def test_watch_samples_all_slots():
    """Inside the context, sample() returns a bounded util for every slot."""
    all_idx = [s.slot_index for s in _SLOTS]
    with gpu_utilization_watch() as sampler:
        util = sampler.sample()
    assert set(util) == set(all_idx), f"missing slots: {set(all_idx) - set(util)}"
    for i, u in util.items():
        assert isinstance(u, int) and 0 <= u <= 100, f"slot {i} util out of range: {u}"


@gpu_only
def test_watch_repeated_samples_are_persistent():
    """Multiple sample() calls in one context each cover all slots (persistent watch)."""
    all_idx = {s.slot_index for s in _SLOTS}
    with gpu_utilization_watch() as sampler:
        a = sampler.sample()
        b = sampler.sample()
    assert set(a) == all_idx and set(b) == all_idx


@gpu_only
def test_watch_subset_of_slots():
    """slot_indices restricts the sample to just those slots."""
    with gpu_utilization_watch([_SLOTS[0].slot_index]) as sampler:
        util = sampler.sample()
    assert set(util) == {_SLOTS[0].slot_index}


@gpu_only
def test_sample_by_pid_shape():
    """sample_by_pid() returns a list per slot; physical=(pid,util) tuples, MIG=[]. """
    with gpu_utilization_watch() as sampler:
        by_pid = sampler.sample_by_pid()
    assert set(by_pid) == {s.slot_index for s in _SLOTS}
    for s in _SLOTS:
        entries = by_pid[s.slot_index]
        assert isinstance(entries, list)
        if s.kind == "mig":
            assert entries == []          # per-MIG per-pid not exposed by GR_ENGINE_ACTIVE
        else:
            for e in entries:
                assert isinstance(e, tuple) and len(e) == 2


@pytest.mark.skipif(
    not (len(_SLOTS) > 0 and os.environ.get("H2O4GPU_MIG_LOAD_TEST") == "1"),
    reason="Set H2O4GPU_MIG_LOAD_TEST=1 and run a GPU load to check >0",
)
def test_watch_nonzero_under_load():
    """With a workload running, at least one slot samples > 0."""
    with gpu_utilization_watch() as sampler:
        util = sampler.sample()
    assert max(util.values()) > 0, f"expected a busy slot to report >0, got {util}"
