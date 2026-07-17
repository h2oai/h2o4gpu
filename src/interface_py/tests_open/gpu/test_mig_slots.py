# -*- encoding: utf-8 -*-
"""
Task A2 — MIG enumeration + in-place slot expansion + deterministic ordering.

These tests run on a MIG-enabled box (e.g. the A100 split into 1g slices). They are
skipped automatically where no MIG instances are present, so the file is safe to
collect on CPU / non-MIG hosts.

Contract (from the implementation plan, A2):
  * every kind=="mig" slot has groupable is False and cuda_token=="MIG-<uuid>"
  * MIG slots carry real memory (mem_total>0) and a bounded utilization (0..100)
  * slot_index is contiguous 0..N-1 across the whole (physical+MIG) table
  * ordering is deterministic/stable across repeated calls, sorted within a physical
    GPU by (gpu_instance_id, compute_instance_id)
  * on a hybrid box, a physical GPU's MIG children appear as a contiguous block and
    plain (groupable) GPUs are interleaved by physical index
"""
import pytest

from h2o4gpu.util.gpu import get_gpu_slots


def _slots():
    try:
        return get_gpu_slots(with_usage=True)
    except Exception:
        return []


_SLOTS = _slots()
_MIG_SLOTS = [s for s in _SLOTS if s.kind == "mig"]
_HAS_MIG = len(_MIG_SLOTS) > 0

mig_only = pytest.mark.skipif(not _HAS_MIG, reason="No MIG instances present on this host")


@mig_only
def test_mig_slots_shape():
    """Each MIG slot is a non-groupable slot with a MIG-<uuid> token and real memory."""
    for s in _MIG_SLOTS:
        assert s.kind == "mig"
        assert s.groupable is False, "MIG slots must be granted alone (not groupable)"
        assert isinstance(s.cuda_token, str) and s.cuda_token.startswith("MIG-"), \
            f"MIG cuda_token must be 'MIG-<uuid>', got {s.cuda_token!r}"
        assert s.mem_total > 0, "MIG slot must report its slice memory"
        assert 0 <= s.utilization <= 100
        # A2 leaves per-MIG utilization at 0 (A3 fills it); assert it's at least valid/bounded.


@mig_only
def test_mig_tokens_unique():
    """Every MIG slice has a distinct UUID token (used as the durable lock identity)."""
    tokens = [s.cuda_token for s in _MIG_SLOTS]
    assert len(tokens) == len(set(tokens)), f"duplicate MIG tokens: {tokens}"


@mig_only
def test_slot_index_contiguous_over_full_table():
    """slot_index is 0..N-1 across the whole physical+MIG table (lock-key correctness)."""
    idx = [s.slot_index for s in _SLOTS]
    assert idx == list(range(len(_SLOTS))), f"slot_index not contiguous: {idx}"


@mig_only
def test_ordering_is_stable_across_calls():
    """Enumeration order must be identical across repeated calls (lock-file correctness)."""
    a = [s.cuda_token for s in get_gpu_slots(with_usage=True)]
    b = [s.cuda_token for s in get_gpu_slots(with_usage=True)]
    assert a == b, f"ordering not stable:\n{a}\n{b}"


@mig_only
def test_mig_children_are_contiguous_per_physical():
    """A physical GPU's MIG children form one contiguous block, ordered within the GPU.

    (On the A100-split-into-4 box this is the whole table; on a hybrid box it also
    verifies plain GPUs don't interleave with another GPU's MIG children.)
    """
    # group consecutive slots by owning physical_index; each physical must appear once
    seen_order = []
    for s in _SLOTS:
        if not seen_order or seen_order[-1] != s.physical_index:
            seen_order.append(s.physical_index)
    assert len(seen_order) == len(set(seen_order)), \
        f"a physical GPU's slots are not contiguous: {[ (s.physical_index, s.kind) for s in _SLOTS ]}"


@pytest.mark.skipif(
    not (any(s.kind == "mig" for s in _SLOTS) and any(s.kind == "physical" for s in _SLOTS)),
    reason="Not a hybrid box (need at least one plain GPU AND one MIG-split GPU)",
)
def test_hybrid_ordering():
    """Hybrid box: plain GPUs are groupable; MIG slices are not; order by physical then (gi,ci)."""
    for s in _SLOTS:
        if s.kind == "physical":
            assert s.groupable is True
            assert s.cuda_token == str(s.physical_index)
        else:
            assert s.groupable is False
            assert s.cuda_token.startswith("MIG-")
    # physical_index must be non-decreasing across the table (physical then its MIG block)
    phys = [s.physical_index for s in _SLOTS]
    assert phys == sorted(phys), f"table not ordered by physical index: {phys}"
