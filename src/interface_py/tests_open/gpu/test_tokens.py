# -*- encoding: utf-8 -*-
"""
Task A5 — slot count, cuda tokens, and MIG-aware cuda_vis_check over the slot table.

Runs on any GPU host (physical or MIG); skipped where no GPU is visible.
"""
import pytest

from h2o4gpu.util.gpu import (
    get_gpu_slots, num_gpu_slots, cuda_token, cuda_tokens, cuda_vis_check,
)

_SLOTS = []
try:
    _SLOTS = get_gpu_slots(with_usage=False)
except Exception:
    pass
gpu_only = pytest.mark.skipif(len(_SLOTS) == 0, reason="No GPU slots available")


@gpu_only
def test_num_gpu_slots_matches_table():
    assert num_gpu_slots() == len(get_gpu_slots(with_usage=False))


@gpu_only
def test_cuda_token_matches_each_slot():
    slots = get_gpu_slots(with_usage=False)
    for s in slots:
        assert cuda_token(s.slot_index) == s.cuda_token


@gpu_only
def test_cuda_tokens_joins_in_order():
    slots = get_gpu_slots(with_usage=False)
    idxs = [s.slot_index for s in slots]
    assert cuda_tokens(idxs) == ",".join(s.cuda_token for s in slots)
    if len(slots) >= 2:
        assert cuda_tokens([0, 1]) == f"{slots[0].cuda_token},{slots[1].cuda_token}"


@gpu_only
def test_cuda_vis_check_unset_sees_all(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    n, idxs = cuda_vis_check()
    slots = get_gpu_slots(with_usage=False)
    assert n == len(slots)
    assert idxs == [s.slot_index for s in slots]


@gpu_only
def test_cuda_vis_check_single_token(monkeypatch):
    """CVD set to one slot's token (int or MIG-<uuid>) -> exactly that slot is visible."""
    target = get_gpu_slots(with_usage=False)[0]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", target.cuda_token)
    n, idxs = cuda_vis_check()
    assert n == 1
    assert idxs == [target.slot_index]


@gpu_only
def test_cuda_vis_check_ignores_unknown_tokens(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-does-not-exist")
    n, idxs = cuda_vis_check()
    assert n == 0 and idxs == []
