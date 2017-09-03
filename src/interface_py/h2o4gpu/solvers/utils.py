# -*- encoding: utf-8 -*-
"""
:copyright: (c) 2017 H2O.ai
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import sys
from ctypes import c_float, c_double
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

def _get_order(data, fortran, order):
    """ Return the Unicode code point representing the
    order of this data set. """
    if data is not None:
        if order is None:
            if fortran:
                order = ord('c')
            else:
                order = ord('r')
        elif order in ['c', 'r']:
            order = ord(order)
        elif order in [ord('c'), ord('r')]:
            order = order
        else:
            ValueError("Bad order")
    return order


def _to_np(data, ismatrix=False, dtype=None, order=None):
    """Convert the input to a numpy array.

    :param data: array_like
    :return: ndarray
    """

    # handle pandas input
    # TODO: Store pandas names at least and attach back to X/coef for output
    import pandas as pd
    if isinstance(data, pd.DataFrame):
        outdata = data.values
    elif isinstance(data, np.ndarray):
        outdata = data
    else:
        outdata = np.asarray(data)

    # deal with degenerate matrices
    if ismatrix and len(outdata.shape) == 1:
        nrows = outdata.shape[0]
        ncols = 1
        outdata = outdata.reshape((nrows, ncols))

    # convert to correct precision if necessary
    if order is not None:
        if order == ord('r'):
            nporder = 'C'
        elif order == ord('c'):
            nporder = 'F'
        else:
            nporder = 'C'
            ValueError("No such order")
    else:
        # in case both (i.e. 1D array), then default to C order
        if outdata.flags.c_contiguous:
            nporder = 'C'
        else:
            nporder = 'F'
    if dtype is None:
        dtype = outdata.dtype

    # force precision as 32-bit if not required types
    if dtype != np.float32 and dtype != np.float64:
        dtype = np.float32

    outdata = outdata.astype(dtype, copy=False, order=nporder)

    selford = _get_order(outdata, fortran=not outdata.flags.c_contiguous,
                         order=order)

    return outdata, selford, dtype


def munge(data_as_np, fit_intercept=False):
    # If True, then append intercept term to
    # train_x array and valid_x array(if available)
    #
    # Not this is really munging as adds to columns
    # and changes expected size of outputted solution
    if (fit_intercept or fit_intercept == 1) and len(data_as_np.shape) == 2:
        data_as_np = np.hstack([data_as_np, np.ones((data_as_np.shape[0], 1),
                                                    dtype=data_as_np.dtype)])
    return data_as_np


def _get_data(data, ismatrix=False, fit_intercept=False,
              order=None, dtype=None):
    """Transforms data to numpy and gather basic info about it.

    :param data: array_like
    :return: data as ndarray, rows, cols, continuity
    """
    # default is no data
    data_as_np = None  # specific to this data
    m = 0  # specific to this data
    n = -1  # specific to this data
    fortran = None  # specific to this data
    # dtype and order not specific to this data, can be just input

    if data is not None:
        data_as_np, order, dtype = _to_np(
            data,
            ismatrix=ismatrix,
            dtype=dtype,
            order=order
        )
        data_as_np = munge(data_as_np, fit_intercept=fit_intercept)
        fortran = not data_as_np.flags.c_contiguous
        shape_x = np.shape(data_as_np)
        m = shape_x[0]
        if len(shape_x) > 1:
            n = shape_x[1]
        else:
            n = 1

    return data_as_np, m, n, fortran, order, dtype


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


def _convert_to_ptr(data):
    """Convert data to a form which can be passed to C/C++ code.

    :param data: array_like
    :return:
    """

    if data is not None:
        np_data, _, dtype = _to_np(data)
        if dtype == np.float32:
            c_ftype = c_float
        elif dtype == np.float64:
            c_ftype = c_double
        else:
            ValueError("No such dtype")
        data_ptr = cptr(np_data, dtype=c_ftype)
    else:
        data_ptr = None

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
