# -*- encoding: utf-8 -*-
"""
:copyright: 2017-2018 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import os
import numpy as np


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


def cuda_vis_check(total_gpus):
    """Helper function to count GPUs by environment variable
    """
    cudavis = os.getenv("CUDA_VISIBLE_DEVICES")
    which_gpus = []
    if cudavis is not None:
        # prune away white-space, non-numerics,
        # except commas for simple checking
        cudavis = "".join(cudavis.split())
        import re
        cudavis = re.sub("[^0-9,]", "", cudavis)

        lencudavis = len(cudavis)
        if lencudavis == 0:
            total_gpus = 0
        else:
            total_gpus = min(
                total_gpus,
                os.getenv("CUDA_VISIBLE_DEVICES").count(",") + 1)
            which_gpus = os.getenv("CUDA_VISIBLE_DEVICES").split(",")
            which_gpus = [int(x) for x in which_gpus]
    else:
        which_gpus = [x for x in range(0, total_gpus)]

    return total_gpus, which_gpus


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
        total_gpus, which_gpus = cuda_vis_check(total_gpus_actual)

        total_mem = \
            min([py3nvml.py3nvml.nvmlDeviceGetMemoryInfo(
                py3nvml.py3nvml.nvmlDeviceGetHandleByIndex(i)).total for i in
                 range(total_gpus)])

        gpu_type = py3nvml.py3nvml.nvmlDeviceGetName \
            (py3nvml.py3nvml.nvmlDeviceGetHandleByIndex(0))

        if return_usage:
            for j in range(total_gpus_actual):
                if j in which_gpus:
                    handle = py3nvml.py3nvml.nvmlDeviceGetHandleByIndex(j)
                    util = py3nvml.py3nvml.nvmlDeviceGetUtilizationRates(handle)
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
        total_gpus, which_gpus = cuda_vis_check(total_gpus_actual)

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
                majors_tmp[j], minors_tmp[j], _ = get_compute_capability_orig(j)

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
