# -*- encoding: utf-8 -*-
"""
:copyright: 2017 H2O.ai, Inc.
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
    available_device_count, _, _ = get_gpu_info_c()

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


def get_gpu_info_c(return_usage=False, return_capability=False,
                   return_all=False, verbose=False):
    """Gets the GPU info from C call

    :return:
        Total number of GPUs and total available memory
         (and optionally GPU usage)
    """
    max_gpus = 128
    total_gpus = 0
    total_gpus_actual = 0
    which_gpus = []
    usages_tmp = np.empty(max_gpus, dtype=np.int32)
    total_mems_tmp = np.empty(max_gpus, dtype=np.uint64)
    free_mems_tmp = np.empty(max_gpus, dtype=np.uint64)
    # This 30 should be same as the gpu type in get_gpu_info_c
    gpu_types_tmp = [' ' * 30 for _ in range(64)]
    majors_tmp = np.empty(max_gpus, dtype=np.int32)
    minors_tmp = np.empty(max_gpus, dtype=np.int32)

    try:
        from ..libs.lib_utils import GPUlib
        lib = GPUlib().get()

        total_gpus_actual = \
            lib.get_gpu_info_c(usages_tmp, total_mems_tmp, free_mems_tmp,
                               gpu_types_tmp, majors_tmp, minors_tmp)

        # This will drop the GPU count, but the returned usage
        total_gpus, which_gpus = cuda_vis_check(total_gpus_actual)

        # Strip the trailing NULL and whitespaces from C backend
        gpu_types_tmp = [g_type.strip().replace("\x00", "")
                         for g_type in gpu_types_tmp]
    # pylint: disable=broad-except
    except Exception as e:
        if verbose:
            print(e)

    total_mems_actual = np.resize(total_mems_tmp, total_gpus_actual)
    free_mems_actual = np.resize(free_mems_tmp, total_gpus_actual)
    gpu_types_actual = np.resize(gpu_types_tmp, total_gpus_actual)
    usages_actual = np.resize(usages_tmp, total_gpus_actual)
    majors_actual = np.resize(majors_tmp, total_gpus_actual)
    minors_actual = np.resize(minors_tmp, total_gpus_actual)

    total_mems = np.resize(np.copy(total_mems_actual), total_gpus)
    free_mems = np.resize(np.copy(free_mems_actual), total_gpus)
    gpu_types = np.resize(np.copy(gpu_types_actual), total_gpus)
    usages = np.resize(np.copy(usages_actual), total_gpus)
    majors = np.resize(np.copy(majors_actual), total_gpus)
    minors = np.resize(np.copy(minors_actual), total_gpus)
    gpu_i = 0
    for j in range(total_gpus_actual):
        if j in which_gpus:
            total_mems[gpu_i] = total_mems_actual[j]
            free_mems[gpu_i] = free_mems_actual[j]
            gpu_types[gpu_i] = gpu_types_actual[j]
            usages[gpu_i] = usages_actual[j]
            gpu_i += 1

    if return_all:
        return (total_gpus, total_mems,
                free_mems, gpu_types, usages, majors, minors)
    if return_usage and return_capability:
        return (total_gpus, total_mems, gpu_types, usages, majors, minors)
    if return_usage:
        return (total_gpus, total_mems, gpu_types, usages)
    if return_capability:
        return (total_gpus, total_mems, gpu_types, majors, minors)
    return (total_gpus, total_mems, gpu_types)


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
    total_gpus, _, _, majors, minors =\
        get_gpu_info_c(return_capability=True)
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
        _, device_major, device_minor, device_ratioperf = \
            lib.get_compute_capability(gpu_id)
    return device_major, device_minor, device_ratioperf
