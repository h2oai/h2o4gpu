# -*- encoding: utf-8 -*-
"""
:copyright: (c) 2017 H2O.ai
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import sys
from ctypes import c_float, POINTER
import numpy as np
from h2o4gpu.types import cptr
from py3nvml.py3nvml import NVMLError

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
            print(
                "Cannot set n_gpus to all GPUs %d %d, trying n_gpus=1"
                % (n_gpus, available_device_count)
            )
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
    try:
        import py3nvml.py3nvml
        py3nvml.py3nvml.nvmlInit()
        total_gpus = py3nvml.py3nvml.nvmlDeviceGetCount()

        total_mem = \
            min([py3nvml.py3nvml.nvmlDeviceGetMemoryInfo(
                py3nvml.py3nvml.nvmlDeviceGetHandleByIndex(i)).total for i in
                 range(total_gpus)])
    except NVMLError as e:
        print("No GPU, setting total_gpus=0 and total_mem=0")
        print(e)
        sys.stdout.flush()
    return total_gpus, total_mem


#############################
# Data utils

def _to_np(data):
    """Convert the input to a numpy array.

    :param data: array_like
    :return: ndarray
    """
    import pandas as pd
    if isinstance(data, pd.DataFrame):
        outdata = data.values
    elif isinstance(data, np.ndarray):
        outdata = data
    else:
        outdata = np.asarray(data)

    return outdata

def munge(data_as_np, fit_intercept = False):
    # If True, then append intercept term to train_x array and valid_x array(if available)
    # Not this is really munging as adds to columns and changes expected size of outputted solution
    if (fit_intercept or fit_intercept == 1) and len(data_as_np.shape) == 2:
        data_as_np = np.hstack([data_as_np, np.ones((data_as_np.shape[0], 1),
                                              dtype=data_as_np.dtype)])
    return data_as_np

def _get_data(data, fit_intercept = False):
    """Transforms data to numpy and gather basic info about it.

    :param data: array_like
    :return: data as ndarray, rows, cols, continuity
    """
    # default is no data
    data_as_np = None
    m = 0
    n = -1
    fortran = None

    if data is not None:
        data_as_np = _to_np(data)
        data_as_np = munge(data_as_np, fit_intercept = fit_intercept)
        fortran = data_as_np.flags.f_contiguous
        shape_x = np.shape(data_as_np)
        m = shape_x[0]
        if len(shape_x) > 1:
            n = shape_x[1]
        else:
            n = 1

    return data_as_np, m, n, fortran


def _check_data_content(do_check, name, data):
    """Makes sure the data contains no infinite or NaN values

    :param do_check: int
        1 perform checks
        != 1 don't perform checks.
    :param name: str
        Name of the object for logging.
    :param data: array_like
        Data to be checked
    :return:
    """
    if do_check == 1:
        assert np.isfinite(data).all(), "%s contains Inf" % name
        assert not np.isnan(data).any(), "%s contains NA" % name


def _data_info(data, verbose=0):
    """Get info about passed data.

    :param data: array_like
    :param verbose: int, optional, default : 0
        Logging level
    :return:
        Data precision (0 or 1), rows, cols
    """
    double_precision = -1
    m = 0
    n = -1

    if data is not None:
        try:
            if data.dtype == np.float64:
                if verbose > 0:
                    print('Detected np.float64 data')
                    sys.stdout.flush()
                double_precision = 1
            if data.dtype == np.float32:
                if verbose > 0:
                    print('Detected np.float32 data')
                    sys.stdout.flush()
                double_precision = 0
        except AttributeError:
            double_precision = -1

        data_shape = np.shape(data)
        if len(data_shape) == 1:
            m = data_shape[0]
        elif len(data_shape) == 2:
            m = data_shape[0]
            n = data_shape[1]

    return double_precision, m, n


def _convert_to_ptr(data, c_ftype=c_float):
    """Convert data to a form which can be passed to C/C++ code.

    :param data: array_like
    :param c_ftype:
    :return:
    """
    data_ptr = POINTER(c_ftype)()

    if data is not None:
        np_data = _to_np(data)
        data_ptr = cptr(np_data, dtype=c_ftype)

    return data_ptr


def _check_equal(iterator):
    """Check if all the values in an iterator are equal.

    :param iterator: iterator
    :return: bool
    """

    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)
