# -*- encoding: utf-8 -*-
"""
:copyright: 2017-2018 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import os
import ctypes as _ct
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# GPU Slot model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProcInfo:
    """Per-process GPU resource usage for one device slot."""
    pid: int
    used_mem: int   # bytes
    usage: int      # percent


@dataclass(frozen=True)
class GpuSlot:
    """
    Unified, ordered descriptor for one GPU slot (physical GPU or MIG instance).
    """
    slot_index: int                    # 0..N-1 — identity / AcquireGPUs lock key
    cuda_token: str                    # CUDA_VISIBLE_DEVICES value: "i" (physical) or "MIG-<uuid>" (mig)
    kind: str                          # "physical" | "mig"
    groupable: bool                    # True=physical (combinable), False=mig (alone)
    name: str
    compute_capability: Tuple[int, int]
    mem_total: int                     # bytes
    mem_used: int                      # bytes  (mem_total - mem_free)
    mem_free: int                      # bytes
    utilization: int                   # percent (0 when with_usage=False)
    physical_index: int                # owning physical GPU (self for physical; parent for mig)
    procs: Optional[List[ProcInfo]]    # per-process info when with_procs=True, else None


def get_gpu_slots(with_usage: bool = True, with_procs: bool = False) -> List[GpuSlot]:
    """
    Return one GpuSlot per visible GPU.

    Tuple shape from get_gpu_info_c when called with the flags below:
        [0]  count                          (int)
        [1]  total_mems  (return_memory=True)
        [2]  gpu_types   (return_name=True)
        [3]  usages      (return_usage=with_usage)
        [4]  free_mems   (return_free_memory=True)
        [5]  majors      (return_capability=True)
        [6]  minors      (return_capability=True)
     if with_procs:
        [7]  num_pids         (return_memory_by_pid=True)
        [8]  pids             shape (count, max_pids)
        [9]  usedGpuMemorys   shape (count, max_pids)
        [10] num_pids_usage   (return_usage_by_pid=True)
        [11] pids_usage       shape (count, max_pids)
        [12] usedGpuUsage     shape (count, max_pids)
    """
    raw = get_gpu_info_c(
        return_memory=True,
        return_name=True,
        return_usage=with_usage,
        return_free_memory=True,
        return_capability=True,
        return_memory_by_pid=with_procs,
        return_usage_by_pid=with_procs,
        return_all=False,
    )

    if raw is None:
        return []

    count = raw[0]
    if count == 0:
        return []

    # Unpack positional tuple elements (indices depend on which flags were set).
    # Flags always set: memory, name, usage (conditional), free_memory, capability.
    # Index 0  = count (always)
    # Index 1  = total_mems (return_memory=True, always)
    # Index 2  = gpu_types  (return_name=True, always)
    # Index 3  = usages     (return_usage=with_usage)
    # Index 4  = free_mems  (return_free_memory=True, always)
    # Index 5  = majors     (return_capability=True, always)
    # Index 6  = minors     (return_capability=True, always)
    # When with_usage=False, indices shift: [3]=free_mems, [4]=majors, [5]=minors
    if with_usage:
        total_mems = raw[1]
        gpu_types  = raw[2]
        usages     = raw[3]
        free_mems  = raw[4]
        majors     = raw[5]
        minors     = raw[6]
        pid_base   = 7
    else:
        total_mems = raw[1]
        gpu_types  = raw[2]
        usages     = None
        free_mems  = raw[3]
        majors     = raw[4]
        minors     = raw[5]
        pid_base   = 6

    # Per-process arrays (only present when with_procs=True)
    if with_procs:
        num_pids_mem      = raw[pid_base]
        pids_mem          = raw[pid_base + 1]   # shape (count, max_pids)
        used_gpu_memorys  = raw[pid_base + 2]   # shape (count, max_pids)
        num_pids_usage    = raw[pid_base + 3]
        pids_usage_arr    = raw[pid_base + 4]   # shape (count, max_pids)
        used_gpu_usage    = raw[pid_base + 5]   # shape (count, max_pids)
    else:
        num_pids_mem = num_pids_usage = None
        pids_mem = used_gpu_memorys = pids_usage_arr = used_gpu_usage = None

    slots: List[GpuSlot] = []
    for i in range(count):
        mem_total_i = int(total_mems[i])
        mem_free_i  = int(free_mems[i])
        mem_used_i  = mem_total_i - mem_free_i

        util_i = int(usages[i]) if usages is not None else 0

        name_i = str(gpu_types[i]) if gpu_types[i] is not None else ""

        cap_i = (int(majors[i]), int(minors[i]))

        if with_procs and num_pids_mem is not None:
            n_mem = int(num_pids_mem[i])
            n_use = int(num_pids_usage[i])
            pid_mem_map = {}
            for k in range(n_mem):
                pid = int(pids_mem[i, k])
                used = int(used_gpu_memorys[i, k])
                pid_mem_map[pid] = [pid, used, 0]
            for k in range(n_use):
                pid = int(pids_usage_arr[i, k])
                usage_val = int(used_gpu_usage[i, k])
                if pid in pid_mem_map:
                    pid_mem_map[pid][2] = usage_val
                else:
                    pid_mem_map[pid] = [pid, 0, usage_val]
            procs_i: Optional[List[ProcInfo]] = [
                ProcInfo(pid=p[0], used_mem=p[1], usage=p[2])
                for p in pid_mem_map.values()
            ]
        else:
            procs_i = None

        slots.append(GpuSlot(
            slot_index=i,
            cuda_token=str(i),
            kind="physical",
            groupable=True,
            name=name_i,
            compute_capability=cap_i,
            mem_total=mem_total_i,
            mem_used=mem_used_i,
            mem_free=mem_free_i,
            utilization=util_i,
            physical_index=i,
            procs=procs_i,
        ))

    return _expand_mig_slots(slots, with_usage=with_usage)


def _expand_mig_slots(physical_slots: List[GpuSlot],
                      with_usage: bool = True) -> List[GpuSlot]:
    """
    Splice MIG instances into the slot table with per-MIG util.
    """
    mig_by_phys = _mig_instances_by_physical()
    if not mig_by_phys:
        return physical_slots

    util_by_gi = _mig_utilization(mig_by_phys) if with_usage else {}

    expanded: List[GpuSlot] = []
    for slot in physical_slots:
        migs = mig_by_phys.get(slot.physical_index)
        if not migs:
            expanded.append(slot)          
            continue
        for m in migs:                     
            util = int(util_by_gi.get((slot.physical_index, m["gi"]), 0))
            expanded.append(GpuSlot(
                slot_index=-1,             
                cuda_token=m["uuid"],    
                kind="mig",
                groupable=False,          
                name=m["name"],
                compute_capability=m["cc"],
                mem_total=m["mem_total"],
                mem_used=m["mem_total"] - m["mem_free"],
                mem_free=m["mem_free"],
                utilization=util,       
                physical_index=slot.physical_index,
                procs=None,
            ))

    return [replace(s, slot_index=i) for i, s in enumerate(expanded)]


def _mig_instances_by_physical() -> Dict[int, List[dict]]:
    """Enumerate MIG instances per physical GPU via the driver's NVML (ctypes).

    Returns ``{physical_index: [mig_dict, ...]}`` sorted by ``(gi, ci)``; empty dict
    when NVML can't load, MIG is off everywhere, or anything goes wrong (fail soft).
    """
    import ctypes

    class _Mem(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong),
                    ("free", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong)]

    NVML_SUCCESS = 0
    NVML_ERROR_NOT_FOUND = 6
    NVML_DEVICE_MIG_ENABLE = 1

    try:
        nvml = ctypes.CDLL("libnvidia-ml.so.1")
    except OSError:
        return {}

    def _u32():
        return ctypes.c_uint(0)

    result: Dict[int, List[dict]] = {}
    if nvml.nvmlInit_v2() != NVML_SUCCESS:
        return {}
    try:
        n = _u32()
        if nvml.nvmlDeviceGetCount_v2(ctypes.byref(n)) != NVML_SUCCESS:
            return {}
        for phys in range(n.value):
            dev = ctypes.c_void_p()
            if nvml.nvmlDeviceGetHandleByIndex_v2(phys, ctypes.byref(dev)) != NVML_SUCCESS:
                continue
            cur, pend = _u32(), _u32()
            if nvml.nvmlDeviceGetMigMode(dev, ctypes.byref(cur), ctypes.byref(pend)) != NVML_SUCCESS:
                continue
            if cur.value != NVML_DEVICE_MIG_ENABLE:
                continue

            maj, minr = ctypes.c_int(0), ctypes.c_int(0)   # MIG instances inherit parent CC
            nvml.nvmlDeviceGetCudaComputeCapability(dev, ctypes.byref(maj), ctypes.byref(minr))

            maxc = _u32()
            if nvml.nvmlDeviceGetMaxMigDeviceCount(dev, ctypes.byref(maxc)) != NVML_SUCCESS:
                continue

            migs: List[dict] = []
            for idx in range(maxc.value):
                mdev = ctypes.c_void_p()
                rv = nvml.nvmlDeviceGetMigDeviceHandleByIndex(dev, idx, ctypes.byref(mdev))
                if rv == NVML_ERROR_NOT_FOUND:
                    continue                      # sparse index — no instance here
                if rv != NVML_SUCCESS:
                    continue

                uuid_buf = ctypes.create_string_buffer(96)
                nvml.nvmlDeviceGetUUID(mdev, uuid_buf, 96)      # "MIG-<uuid>"
                name_buf = ctypes.create_string_buffer(96)
                nvml.nvmlDeviceGetName(mdev, name_buf, 96)
                mem = _Mem()
                nvml.nvmlDeviceGetMemoryInfo(mdev, ctypes.byref(mem))
                gi, ci = _u32(), _u32()
                nvml.nvmlDeviceGetGpuInstanceId(mdev, ctypes.byref(gi))
                nvml.nvmlDeviceGetComputeInstanceId(mdev, ctypes.byref(ci))

                migs.append({
                    "uuid": uuid_buf.value.decode("utf-8", "replace"),
                    "name": name_buf.value.decode("utf-8", "replace"),
                    "mem_total": int(mem.total),
                    "mem_free": int(mem.free),
                    "cc": (int(maj.value), int(minr.value)),
                    "gi": int(gi.value),
                    "ci": int(ci.value),
                })

            if migs:
                migs.sort(key=lambda m: (m["gi"], m["ci"]))
                result[phys] = migs
    finally:
        nvml.nvmlShutdown()
    return result


def _mig_utilization(mig_by_phys: Dict[int, List[dict]]) -> Dict[Tuple[int, int], int]:
    """
    Per-MIG utilization keyed by ``(physical_index, gpu_instance_id)``.
    """
    # TODO(B200): if nvmlGpmQueryDeviceSupport() is True for a MIG parent, use NVML GPM
    #             for that GPU instead of DCGM. Until B200 hardware is available we take
    #             the DCGM (Ampere) branch for every MIG-enabled GPU.
    return _mig_utilization_dcgm(mig_by_phys)


# ---------------------------------------------------------------------------
# DCGM client for per-MIG utilization (GR_ENGINE_ACTIVE) — Ampere path.
# Struct layouts / versions / enum values mirror the vendored headers in
# third_party/dcgm (dcgm_structs.h, dcgm_fields.h, dcgm_agent.h).
# ---------------------------------------------------------------------------

_DCGM_ST_OK = 0
_DCGM_FE_GPU_I = 4                        # dcgm_field_entity_group_t
_DCGM_FI_PROF_GR_ENGINE_ACTIVE = 1001
_DCGM_GROUP_DEFAULT_INSTANCES = 3         # dcgmGroupType_t: all GPU instances
_DCGM_MAX_BLOB_LENGTH = 4096
_DCGM_MAX_HIERARCHY_INFO = 32 * 14        # DCGM_MAX_NUM_DEVICES * DCGM_MAX_TOTAL_INSTANCES_PER_GPU


def _dcgm_ver(struct_type, v):
    return _ct.sizeof(struct_type) | (v << 24)   # MAKE_DCGM_VERSION


class _DcgmConnectParams(_ct.Structure):
    _fields_ = [("version", _ct.c_uint), ("persistAfterDisconnect", _ct.c_uint),
                ("timeoutMs", _ct.c_uint), ("addressIsUnixSocket", _ct.c_uint)]


class _DcgmEntityPair(_ct.Structure):
    _fields_ = [("entityGroupId", _ct.c_int), ("entityId", _ct.c_uint)]


class _DcgmFVUnion(_ct.Union):
    _fields_ = [("i64", _ct.c_int64), ("dbl", _ct.c_double),
                ("blob", _ct.c_char * _DCGM_MAX_BLOB_LENGTH)]


class _DcgmFieldValue(_ct.Structure):
    _fields_ = [("version", _ct.c_uint), ("entityGroupId", _ct.c_int),
                ("entityId", _ct.c_uint), ("fieldId", _ct.c_ushort),
                ("fieldType", _ct.c_ushort), ("status", _ct.c_int),
                ("unused", _ct.c_uint), ("ts", _ct.c_int64), ("value", _DcgmFVUnion)]


class _DcgmMigEntityInfo(_ct.Structure):
    _fields_ = [("gpuUuid", _ct.c_char * 128), ("nvmlGpuIndex", _ct.c_uint),
                ("nvmlInstanceId", _ct.c_uint), ("nvmlComputeInstanceId", _ct.c_uint),
                ("nvmlMigProfileId", _ct.c_uint), ("nvmlProfileSlices", _ct.c_uint)]


class _DcgmMigHierInfo(_ct.Structure):
    _fields_ = [("entity", _DcgmEntityPair), ("parent", _DcgmEntityPair),
                ("info", _DcgmMigEntityInfo)]


class _DcgmMigHierarchy(_ct.Structure):
    _fields_ = [("version", _ct.c_uint), ("count", _ct.c_uint),
                ("entityList", _DcgmMigHierInfo * _DCGM_MAX_HIERARCHY_INFO)]


def _load_dcgm():
    """Load the bundled libdcgm.so.4.5.3 (host libdcgm.so.4 fallback); None if absent."""
    import glob
    libdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib")
    for cand in sorted(glob.glob(os.path.join(libdir, "libdcgm.so*"))) + ["libdcgm.so.4"]:
        try:
            return _ct.CDLL(cand)
        except OSError:
            continue
    return None


class _DcgmMigSession:
    """
    Persistent DCGM connection + GR_ENGINE_ACTIVE watch on the GPU-instances group.

    ``open()`` connects to the standalone nv-hostengine and starts the watch once;
    ``sample()`` then updates+reads cheaply (no reconnect); ``close()`` tears it down.
    """

    def __init__(self):
        self._dcgm = None
        self._handle = _ct.c_uint64(0)
        self._connected = False
        self.entity_to_key = {}          # dcgm GPU_I entityId -> (physical_index, gi)

    def open(self):
        dcgm = _load_dcgm()
        if dcgm is None:
            return False
        from ctypes import (c_int, c_uint, c_uint64, c_longlong, c_double, c_ushort,
                            c_char_p, POINTER)
        dcgm.dcgmConnect_v2.argtypes = [c_char_p, POINTER(_DcgmConnectParams), POINTER(c_uint64)]
        dcgm.dcgmGetGpuInstanceHierarchy.argtypes = [c_uint64, POINTER(_DcgmMigHierarchy)]
        dcgm.dcgmFieldGroupCreate.argtypes = [c_uint64, c_int, POINTER(c_ushort), c_char_p,
                                              POINTER(c_uint64)]
        dcgm.dcgmGroupCreate.argtypes = [c_uint64, c_int, c_char_p, POINTER(c_uint64)]
        dcgm.dcgmWatchFields.argtypes = [c_uint64, c_uint64, c_uint64, c_longlong, c_double, c_int]
        dcgm.dcgmUpdateAllFields.argtypes = [c_uint64, c_int]
        dcgm.dcgmEntitiesGetLatestValues.argtypes = [c_uint64, POINTER(_DcgmEntityPair), c_uint,
                                                     POINTER(c_ushort), c_uint, c_uint,
                                                     POINTER(_DcgmFieldValue)]
        self._dcgm = dcgm
        if dcgm.dcgmInit() != _DCGM_ST_OK:
            self._dcgm = None
            return False
        address = os.environ.get("H2O4GPU_DCGM_DAEMON_ADDRESS", "127.0.0.1").encode("ascii")
        params = _DcgmConnectParams(version=_dcgm_ver(_DcgmConnectParams, 2),
                                    persistAfterDisconnect=0, timeoutMs=5000,
                                    addressIsUnixSocket=0)
        if dcgm.dcgmConnect_v2(address, _ct.byref(params), _ct.byref(self._handle)) != _DCGM_ST_OK:
            self.close()
            return False
        self._connected = True
        try:
            hier = _DcgmMigHierarchy(version=_dcgm_ver(_DcgmMigHierarchy, 2))
            if dcgm.dcgmGetGpuInstanceHierarchy(self._handle, _ct.byref(hier)) != _DCGM_ST_OK:
                self.close()
                return False
            for k in range(hier.count):
                e = hier.entityList[k]
                if e.entity.entityGroupId == _DCGM_FE_GPU_I:
                    self.entity_to_key[int(e.entity.entityId)] = (int(e.info.nvmlGpuIndex),
                                                                  int(e.info.nvmlInstanceId))
            if not self.entity_to_key:
                self.close()
                return False
            fids = (c_ushort * 1)(_DCGM_FI_PROF_GR_ENGINE_ACTIVE)
            field_group = c_uint64(0)
            if dcgm.dcgmFieldGroupCreate(self._handle, 1, fids, b"h2o4gpu_mig_gract",
                                         _ct.byref(field_group)) != _DCGM_ST_OK:
                self.close()
                return False
            group = c_uint64(0)
            if dcgm.dcgmGroupCreate(self._handle, _DCGM_GROUP_DEFAULT_INSTANCES, b"h2o4gpu_mig",
                                    _ct.byref(group)) != _DCGM_ST_OK:
                self.close()
                return False
            # 100ms updates; groups/watches auto-clean on dcgmDisconnect (persist=0).
            if dcgm.dcgmWatchFields(self._handle, group, field_group,
                                    100000, 5.0, 100) != _DCGM_ST_OK:
                self.close()
                return False
            dcgm.dcgmUpdateAllFields(self._handle, 1)
            return True
        except Exception:
            self.close()
            return False

    def sample(self):
        """{(physical_index, gi): util%} from a fresh read of the watched GR_ENGINE_ACTIVE."""
        if not self._connected or not self.entity_to_key:
            return {}
        from ctypes import c_ushort
        dcgm = self._dcgm
        try:
            dcgm.dcgmUpdateAllFields(self._handle, 1)
            ids = sorted(self.entity_to_key)
            n = len(ids)
            entities = (_DcgmEntityPair * n)(*[_DcgmEntityPair(_DCGM_FE_GPU_I, i) for i in ids])
            fields = (c_ushort * 1)(_DCGM_FI_PROF_GR_ENGINE_ACTIVE)
            values = (_DcgmFieldValue * n)()
            for v in values:
                v.version = _dcgm_ver(_DcgmFieldValue, 2)
            if dcgm.dcgmEntitiesGetLatestValues(self._handle, entities, n, fields,
                                                1, 0, values) != _DCGM_ST_OK:
                return {}
        except Exception:
            return {}
        out = {}
        for j in range(n):
            v = values[j]
            if v.status != _DCGM_ST_OK:
                continue
            key = self.entity_to_key.get(int(v.entityId))
            if key is None:
                continue
            frac = v.value.dbl                    # GR_ENGINE_ACTIVE is a 0.0..1.0 ratio
            if frac != frac or frac < 0:          # NaN / blank sample -> treat as 0
                frac = 0.0
            out[key] = int(round(min(1.0, frac) * 100))
        return out

    def close(self):
        dcgm = self._dcgm
        if dcgm is None:
            return
        if self._connected:
            try:
                dcgm.dcgmDisconnect(self._handle)
            except Exception:
                pass
            self._connected = False
        try:
            dcgm.dcgmShutdown()
        except Exception:
            pass
        self._dcgm = None


def _mig_utilization_dcgm(mig_by_phys: Dict[int, List[dict]]) -> Dict[Tuple[int, int], int]:
    import time
    session = _DcgmMigSession()
    if not session.open():
        return {}
    try:
        time.sleep(0.3)                           # let the profiling module post a sample
        return session.sample()
    finally:
        session.close()


# ---------------------------------------------------------------------------
# gpu_utilization_watch() — persistent, low-overhead sampler
# ---------------------------------------------------------------------------

def _physical_utilization() -> Dict[int, int]:
    """{physical_index: util%} for physical GPUs (NVML, via get_gpu_info_c)."""
    raw = get_gpu_info_c(return_usage=True)
    if not raw or raw[0] == 0:
        return {}
    usages = raw[1]
    return {i: int(usages[i]) for i in range(raw[0])}


def _physical_utilization_by_pid() -> Dict[int, List[Tuple[int, int]]]:
    """{physical_index: [(pid, util%), ...]} for physical GPUs (via get_gpu_info_c)."""
    raw = get_gpu_info_c(return_usage_by_pid=True)
    if not raw or raw[0] == 0:
        return {}
    count, num, pids, used = raw[0], raw[1], raw[2], raw[3]
    out: Dict[int, List[Tuple[int, int]]] = {}
    for i in range(count):
        m = int(num[i])
        out[i] = [(int(pids[i, k]), int(used[i, k])) for k in range(m)]
    return out


class UtilSampler:
    """
    Repeated GPU-utilization sampler over the slot table.
    """

    def __init__(self, slots, requested, uuid_to_key, dcgm_session):
        self._by_index = {s.slot_index: s for s in slots}
        self._requested = [i for i in requested if i in self._by_index]
        self._uuid_to_key = uuid_to_key           # MIG uuid -> (physical_index, gi)
        self._dcgm = dcgm_session

    def sample(self) -> Dict[int, int]:
        """{slot_index: util%} for the requested slots."""
        want = [self._by_index[i] for i in self._requested]
        need_mig = any(s.kind == "mig" for s in want)
        need_phys = any(s.kind == "physical" for s in want)
        mig_util = self._dcgm.sample() if (need_mig and self._dcgm is not None) else {}
        phys_util = _physical_utilization() if need_phys else {}
        out: Dict[int, int] = {}
        for s in want:
            if s.kind == "mig":
                key = self._uuid_to_key.get(s.cuda_token)
                out[s.slot_index] = int(mig_util.get(key, 0)) if key is not None else 0
            else:
                out[s.slot_index] = int(phys_util.get(s.physical_index, 0))
        return out

    def sample_by_pid(self) -> Dict[int, List[Tuple[int, int]]]:
        """{slot_index: [(pid, util%), ...]}.

        Physical slots report per-process usage (via NVML). MIG slots report ``[]``:
        GR_ENGINE_ACTIVE is per-GPU-instance, not per-process, so per-MIG per-pid
        utilization needs DCGM process accounting (deferred).
        """
        want = [self._by_index[i] for i in self._requested]
        phys_pid = (_physical_utilization_by_pid()
                    if any(s.kind == "physical" for s in want) else {})
        out: Dict[int, List[Tuple[int, int]]] = {}
        for s in want:
            out[s.slot_index] = (phys_pid.get(s.physical_index, [])
                                 if s.kind == "physical" else [])
        return out


@contextmanager
def gpu_utilization_watch(slot_indices=None):
    """
    Yield a :class:`UtilSampler` for repeated, low-overhead sampling.
    """
    slots = get_gpu_slots(with_usage=False)
    by_index = {s.slot_index: s for s in slots}
    requested = ([s.slot_index for s in slots] if slot_indices is None
                 else [i for i in slot_indices if i in by_index])
    uuid_to_key = {m["uuid"]: (phys, m["gi"])
                   for phys, migs in _mig_instances_by_physical().items() for m in migs}
    need_mig = any(by_index[i].kind == "mig" for i in requested)
    session = None
    if need_mig:
        session = _DcgmMigSession()
        if not session.open():
            session = None
    try:
        yield UtilSampler(slots, requested, uuid_to_key, session)
    finally:
        if session is not None:
            session.close()


#############################
# Device utils


def device_count(n_gpus=0):
    """Tries to return the number of available GPUs on this machine.

    :param n_gpus: int, optional, default : 0
        If < 0 then return all available GPUs
        If >= 0 then return n_gpus or as many as possible
    :return:
        Adjusted n_gpus and all available devices
    """
    available_device_count = get_gpu_info_c()[0]

    if n_gpus < 0:
        if available_device_count >= 0:
            n_gpus = available_device_count
        else:
            print("Cannot set n_gpus to all GPUs %d %d, trying n_gpus=1" %
                  (n_gpus, available_device_count))
            n_gpus = 1

    if n_gpus > available_device_count:
        n_gpus = available_device_count

    return n_gpus, available_device_count


def get_gpu_info(return_usage=False, trials=2, timeout=30, print_trials=False):
    """Gets the GPU info.

    This runs in a sub-process to avoid mixing parent-child CUDA contexts.
    # get GPU info, but do in sub-process
    # to avoid mixing parent-child cuda contexts
    # https://stackoverflow.com/questions/22950047/cuda-initialization-error-after-fork
    # Tries "trials" times to get result
    # If fails to get result within "timeout" seconds each trial,
    #    then returns as if no GPU

    :return:
        Total number of GPUs and total available memory
    """
    total_gpus = 0
    total_mem = 0
    gpu_type = 0
    usage = []
    import concurrent.futures
    from concurrent.futures import ProcessPoolExecutor
    res = None
    # sometimes hit broken process pool in cpu mode,
    # so just return back no gpus.
    for trial in range(0, trials):
        try:
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(get_gpu_info_subprocess, return_usage)
                # don't wait more than 30s,
                # import on py3nvml can hang if 2 subprocesses
                # GIL lock import at same time
                res = future.result(timeout=timeout)
            return res
        except concurrent.futures.process.BrokenProcessPool:
            pass
        except concurrent.futures.TimeoutError:
            pass
        if print_trials:
            print("Trial %d/%d" % (trial, trials - 1))
    if return_usage:
        return (total_gpus, total_mem, gpu_type, usage)
    return (total_gpus, total_mem, gpu_type)


def _cuda_vis_check_physical(total_gpus):
    """Physical-device visibility filter used by get_gpu_info_c (integer CVD tokens).

    MIG-<uuid> tokens are ignored here (they are not physical device indices); if
    CUDA_VISIBLE_DEVICES is all-MIG, every physical device stays visible so NVML can
    still enumerate the parent(s) for MIG expansion.
    """
    cudavis = os.getenv("CUDA_VISIBLE_DEVICES")
    if cudavis is None:
        return total_gpus, list(range(total_gpus))
    tokens = [t for t in "".join(cudavis.split()).split(",") if t]
    if not tokens:                       # CUDA_VISIBLE_DEVICES="" -> nothing visible
        return 0, []
    int_tokens = [int(t) for t in tokens if t.lstrip("+-").isdigit()]
    if not int_tokens:                   # all MIG-<uuid> tokens -> keep parents visible
        return total_gpus, list(range(total_gpus))
    return min(total_gpus, len(int_tokens)), int_tokens


# ---------------------------------------------------------------------------
# slot count + cuda tokens + MIG-aware visibility (over the slot table)
# ---------------------------------------------------------------------------

def num_gpu_slots():
    """Total GPU slots (physical GPUs + MIG instances) — DAI's ngpus_vis."""
    return len(get_gpu_slots(with_usage=False))


def cuda_token(slot_index):
    """CUDA_VISIBLE_DEVICES token for one slot: "<i>" (physical) or "MIG-<uuid>" (mig)."""
    return get_gpu_slots(with_usage=False)[slot_index].cuda_token


def cuda_tokens(slot_indices):
    """Comma-joined CUDA_VISIBLE_DEVICES string for the given slot indices."""
    slots = get_gpu_slots(with_usage=False)
    return ",".join(slots[i].cuda_token for i in slot_indices)


def cuda_vis_check(total_slots=None):
    """MIG-aware slot visibility per CUDA_VISIBLE_DEVICES.

    Returns ``(count, [slot_index, ...])`` for the slots CVD exposes, honoring both
    integer tokens (physical slots) and ``MIG-<uuid>`` tokens, matched against each
    slot's ``cuda_token``.  When CVD is unset, all slots are visible.  ``total_slots``
    is accepted for signature compatibility; the slot table is the source of truth.
    """
    slots = get_gpu_slots(with_usage=False)
    token_to_index = {s.cuda_token: s.slot_index for s in slots}
    cudavis = os.getenv("CUDA_VISIBLE_DEVICES")
    if cudavis is None:
        idxs = [s.slot_index for s in slots]
        return len(idxs), idxs
    tokens = [t for t in "".join(cudavis.split()).split(",") if t]
    idxs = [token_to_index[t] for t in tokens if t in token_to_index]
    return len(idxs), idxs


def get_gpu_info_subprocess(return_usage=False):
    """Gets the GPU info in a subprocess

    :return:
        Total number of GPUs and total available memory
         (and  optionally GPU usage)
    """
    total_gpus = 0
    total_mem = 0
    gpu_type = 0
    usage = []
    try:
        import py3nvml.py3nvml
        py3nvml.py3nvml.nvmlInit()
        total_gpus_actual = py3nvml.py3nvml.nvmlDeviceGetCount()

        # the below restricts but doesn't select
        total_gpus, which_gpus = _cuda_vis_check_physical(total_gpus_actual)

        total_mem = \
            min([py3nvml.py3nvml.nvmlDeviceGetMemoryInfo(
                py3nvml.py3nvml.nvmlDeviceGetHandleByIndex(i)).total
                 for i in range(total_gpus)])

        gpu_type = py3nvml.py3nvml.nvmlDeviceGetName(
            py3nvml.py3nvml.nvmlDeviceGetHandleByIndex(0))

        if return_usage:
            for j in range(total_gpus_actual):
                if j in which_gpus:
                    handle = py3nvml.py3nvml.nvmlDeviceGetHandleByIndex(j)
                    util = py3nvml.py3nvml.nvmlDeviceGetUtilizationRates(
                        handle)
                    usage.append(util.gpu)
    # pylint: disable=bare-except
    except:
        pass

    if return_usage:
        return (total_gpus, total_mem, gpu_type, usage)
    return (total_gpus, total_mem, gpu_type)


def get_gpu_info_c(return_memory=False,
                   return_name=False,
                   return_usage=False,
                   return_free_memory=False,
                   return_capability=False,
                   return_memory_by_pid=False,
                   return_usage_by_pid=False,
                   return_all=False,
                   verbose=0):
    """Gets the GPU info from C call

    :return:
        Total number of GPUs and total available memory
         (and optionally GPU usage)
    """

    # For backwards compatibility
    # Don't change to `if verbose:` it will catch also int values > 0
    if verbose is True:
        verbose = 600
    if verbose is False:
        verbose = 0

    max_gpus = 16
    total_gpus = 0
    total_gpus_actual = 0
    which_gpus = []
    usages_tmp = np.zeros(max_gpus, dtype=np.int32)
    total_mems_tmp = np.zeros(max_gpus, dtype=np.uint64)
    free_mems_tmp = np.zeros(max_gpus, dtype=np.uint64)
    # This 100 should be same as the gpu type in get_gpu_info_c
    gpu_types_tmp = [' ' * 100 for _ in range(max_gpus)]
    majors_tmp = np.zeros(max_gpus, dtype=np.int32)
    minors_tmp = np.zeros(max_gpus, dtype=np.int32)
    max_pids = 2000
    num_pids_tmp = np.zeros(max_pids, dtype=np.uint32)
    pids_tmp = np.zeros(max_pids * max_gpus, dtype=np.uint32)
    usedGpuMemorys_tmp = np.zeros(max_pids * max_gpus, dtype=np.uint64)
    num_pids_usage_tmp = np.zeros(max_pids, dtype=np.uint32)
    pids_usage_tmp = np.zeros(max_pids * max_gpus, dtype=np.uint32)
    usedGpuUsage_tmp = np.zeros(max_pids * max_gpus, dtype=np.uint64)

    try:
        from ..libs.lib_utils import GPUlib
        lib = GPUlib().get(verbose=verbose)

        status, total_gpus_actual = \
            lib.get_gpu_info_c(verbose,
                               1 if return_memory else 0,
                               1 if return_name else 0,
                               1 if return_usage else 0,
                               1 if return_free_memory else 0,
                               1 if return_capability else 0,
                               1 if return_memory_by_pid else 0,
                               1 if return_usage_by_pid else 0,
                               1 if return_all else 0,
                               usages_tmp, total_mems_tmp, free_mems_tmp,
                               gpu_types_tmp, majors_tmp, minors_tmp,
                               num_pids_tmp, pids_tmp, usedGpuMemorys_tmp,
                               num_pids_usage_tmp, pids_usage_tmp,
                               usedGpuUsage_tmp)

        if status != 0:
            return None

        # This will drop the GPU count, but the returned usage
        total_gpus, which_gpus = _cuda_vis_check_physical(total_gpus_actual)

        # Strip the trailing NULL and whitespaces from C backend
        gpu_types_tmp = [g_type.strip().replace("\x00", "")
                         for g_type in gpu_types_tmp]
    # pylint: disable=broad-except
    except Exception as e:
        if verbose > 0:
            import sys
            sys.stderr.write("Exception: %s" % str(e))
            print(e)
            sys.stdout.flush()

    if return_capability or return_all:
        if list(minors_tmp)[0] == -1:
            for j in which_gpus:
                majors_tmp[j], minors_tmp[j], _ = get_compute_capability_orig(
                    j)

    total_mems_actual = np.resize(total_mems_tmp, total_gpus_actual)
    free_mems_actual = np.resize(free_mems_tmp, total_gpus_actual)
    gpu_types_actual = np.resize(gpu_types_tmp, total_gpus_actual)
    usages_actual = np.resize(usages_tmp, total_gpus_actual)
    majors_actual = np.resize(majors_tmp, total_gpus_actual)
    minors_actual = np.resize(minors_tmp, total_gpus_actual)
    num_pids_actual = np.resize(num_pids_tmp, total_gpus_actual)
    pids_actual = np.resize(pids_tmp, total_gpus_actual * max_pids)
    usedGpuMemorys_actual = np.resize(usedGpuMemorys_tmp,
                                      total_gpus_actual * max_pids)
    num_pids_usage_actual = np.resize(num_pids_usage_tmp, total_gpus_actual)
    pids_usage_actual = np.resize(pids_usage_tmp, total_gpus_actual * max_pids)
    usedGpuUsage_actual = np.resize(usedGpuUsage_tmp,
                                    total_gpus_actual * max_pids)

    total_mems = np.resize(np.copy(total_mems_actual), total_gpus)
    free_mems = np.resize(np.copy(free_mems_actual), total_gpus)
    gpu_types = np.resize(np.copy(gpu_types_actual), total_gpus)
    usages = np.resize(np.copy(usages_actual), total_gpus)
    majors = np.resize(np.copy(majors_actual), total_gpus)
    minors = np.resize(np.copy(minors_actual), total_gpus)
    num_pids = np.resize(np.copy(num_pids_actual), total_gpus)
    pids = np.resize(np.copy(pids_actual), total_gpus * max_pids)
    usedGpuMemorys = np.resize(np.copy(usedGpuMemorys_actual),
                               total_gpus * max_pids)
    num_pids_usage = np.resize(np.copy(num_pids_usage_actual), total_gpus)
    pids_usage = np.resize(np.copy(pids_usage_actual), total_gpus * max_pids)
    usedGpuUsage = np.resize(np.copy(usedGpuUsage_actual),
                             total_gpus * max_pids)

    gpu_i = 0
    for j in range(total_gpus_actual):
        if j in which_gpus:
            total_mems[gpu_i] = total_mems_actual[j]
            free_mems[gpu_i] = free_mems_actual[j]
            gpu_types[gpu_i] = gpu_types_actual[j]
            usages[gpu_i] = usages_actual[j]
            minors[gpu_i] = minors_actual[j]
            majors[gpu_i] = majors_actual[j]
            num_pids[gpu_i] = num_pids_actual[j]
            pids[gpu_i] = pids_actual[j]
            usedGpuMemorys[gpu_i] = usedGpuMemorys_actual[j]
            num_pids_usage[gpu_i] = num_pids_usage_actual[j]
            pids_usage[gpu_i] = pids_usage_actual[j]
            usedGpuUsage[gpu_i] = usedGpuUsage_actual[j]
            gpu_i += 1
    pids = np.reshape(pids, (total_gpus, max_pids))
    usedGpuMemorys = np.reshape(usedGpuMemorys, (total_gpus, max_pids))
    pids_usage = np.reshape(pids_usage, (total_gpus, max_pids))
    usedGpuUsage = np.reshape(usedGpuUsage, (total_gpus, max_pids))

    to_return = [total_gpus]
    if return_all or return_memory:
        to_return.append(total_mems)
    if return_all or return_name:
        to_return.append(gpu_types)
    if return_all or return_usage:
        to_return.append(usages)
    if return_all or return_free_memory:
        to_return.append(free_mems)
    if return_all or return_capability:
        to_return.extend([majors, minors])
    if return_all or return_memory_by_pid:
        to_return.extend([num_pids, pids, usedGpuMemorys])
    if return_all or return_usage_by_pid:
        to_return.extend([num_pids_usage, pids_usage, usedGpuUsage])

    return tuple(to_return)


def cudaresetdevice(gpu_id, n_gpus):
    """
    Resets the cuda device so any next cuda call will reset the cuda context.

    :param gpuU_id: int
        device number of GPU (to start with if n_gpus>1)
    :param n_gpus: int, optional, default : 0
        If < 0 then apply to all available GPUs
        If >= 0 then apply to that number of GPUs
    """
    (n_gpus, devices) = device_count(n_gpus)
    gpu_id = gpu_id % devices

    from ..libs.lib_utils import get_lib
    lib = get_lib(n_gpus, devices)
    if lib is None:
        n_gpus = 0

    if n_gpus > 0 and lib is not None:
        lib.cudaresetdevice(gpu_id, n_gpus)


def cudaresetdevice_bare(n_gpus):
    """
    Resets the cuda device so any next cuda call will reset the cuda context.
    """
    if n_gpus > 0:
        from ..libs.lib_utils import GPUlib
        GPUlib().get().cudaresetdevice_bare()


def get_compute_capability(gpu_id):
    """
    Get compute capability for all gpus
    """
    try:
        total_gpus, majors, minors =\
            get_gpu_info_c(return_capability=True)
    # pylint: disable=bare-except
    except:
        total_gpus = 0
    if total_gpus > 0:
        gpu_id = gpu_id % total_gpus
        device_major = majors.tolist()[gpu_id]
        device_minor = minors.tolist()[gpu_id]
        device_ratioperf = 1
    else:
        device_major = -1
        device_minor = -1
        device_ratioperf = 1
    return (device_major, device_minor, device_ratioperf)


def get_compute_capability_orig(gpu_id):
    """
    Gets the major cuda version, minor cuda version,
     and ratio of floating point single perf to double perf.

    :param gpuU_id: int
        device number of GPU
    """
    device_major = -1
    device_minor = -1
    device_ratioperf = 1
    import concurrent.futures
    from concurrent.futures import ProcessPoolExecutor
    res = None
    # sometimes hit broken process pool in cpu mode,
    # so return dummy values in that case
    try:
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(get_compute_capability_subprocess, gpu_id)
            res = future.result()
        return res
    except concurrent.futures.process.BrokenProcessPool:
        return (device_major, device_minor, device_ratioperf)


def get_compute_capability_subprocess(gpu_id):
    """
    Gets the major cuda version, minor cuda version,
     and ratio of floating point single perf to double perf.

    :param gpuU_id: int
        device number of GPU
    """
    n_gpus = -1
    (n_gpus, devices) = device_count(n_gpus)
    gpu_id = gpu_id % devices

    from ..libs.lib_utils import get_lib
    lib = get_lib(n_gpus, devices)
    if lib is None:
        n_gpus = 0

    device_major = 0
    device_minor = 0
    device_ratioperf = 0
    if n_gpus > 0 and lib is not None:
        error, device_major, device_minor, device_ratioperf = \
            lib.get_compute_capability(gpu_id)
        assert error == 0, "Error in get_compute_capability_subprocess"
    return device_major, device_minor, device_ratioperf
