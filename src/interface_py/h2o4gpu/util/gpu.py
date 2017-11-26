# -*- encoding: utf-8 -*-
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""

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
    available_device_count, _ = gpu_info()

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


def gpu_info():
    """Gets the GPU info.

    This runs in a sub-process to avoid mixing parent-child CUDA contexts.

    :return:
        Total number of GPUs and total available memory
    """
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_gpu_info_subprocess)
        res = future.result()
    return res


def _gpu_info_subprocess():
    """Gets the GPU info.

    :return:
        Total number of GPUs and total available memory
    """
    total_gpus = 0
    total_mem = 0
    from py3nvml.py3nvml import NVMLError
    try:
        import py3nvml.py3nvml
        py3nvml.py3nvml.nvmlInit()
        total_gpus = py3nvml.py3nvml.nvmlDeviceGetCount()

        import os
        cudavis = os.getenv("CUDA_VISIBLE_DEVICES")
        if cudavis is not None:
            lencudavis = len(cudavis)
            if lencudavis == 0:
                total_gpus = 0
            else:
                total_gpus =\
                    min(total_gpus,
                        os.getenv("CUDA_VISIBLE_DEVICES").count(",") + 1)

        total_mem = \
            min([py3nvml.py3nvml.nvmlDeviceGetMemoryInfo(
                py3nvml.py3nvml.nvmlDeviceGetHandleByIndex(i)).total for i in
                 range(total_gpus)])
    except NVMLError as e:
        print("No GPU, setting total_gpus=0 and total_mem=0")
        print(e)
        import sys
        sys.stdout.flush()
    return total_gpus, total_mem


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

    from ..libs.lib_elastic_net import GPUlib, CPUlib
    gpu_lib = GPUlib().get()
    cpu_lib = CPUlib().get()

    lib = None
    if n_gpus == 0 or gpu_lib is None or devices == 0:
        lib = cpu_lib
    elif n_gpus > 0 or gpu_lib is None or devices == 0:
        lib = gpu_lib
    else:
        n_gpus = 0

    if n_gpus > 0 and lib is not None:
        from ctypes import c_int
        lib.cudaresetdevice(c_int(gpu_id), c_int(n_gpus))


def get_compute_capability(gpu_id):
    """
    Gets the major cuda version, minor cuda version, and ratio of floating point single perf to double perf.

    :param gpuU_id: int
        device number of GPU
    """
    n_gpus = -1
    (n_gpus, devices) = device_count(n_gpus)
    gpu_id = gpu_id % devices

    from ..libs.lib_elastic_net import GPUlib, CPUlib
    gpu_lib = GPUlib().get()
    cpu_lib = CPUlib().get()

    lib = None
    if n_gpus == 0 or gpu_lib is None or devices == 0:
        lib = cpu_lib
    elif n_gpus > 0 or gpu_lib is None or devices == 0:
        lib = gpu_lib
    else:
        n_gpus = 0

    from ctypes import c_int, c_float, c_double, c_void_p, c_size_t, POINTER, \
        pointer, cast, addressof
    device_major = c_int(0)
    device_minor = c_int(0)
    device_ratioperf = c_int(0)
    if n_gpus > 0 and lib is not None:
        from ctypes import c_int
        c_int_p = POINTER(c_int)
        lib.get_compute_capability(c_int(gpu_id), cast(addressof(device_major),c_int_p), cast(addressof(device_minor),c_int_p), cast(addressof(device_ratioperf),c_int_p))
        #print("device_major=%d device_minor=%d device_ratioperf=%d" % (device_major.value, device_minor.value, device_ratioperf.value))
    return device_major.value, device_minor.value, device_ratioperf.value
