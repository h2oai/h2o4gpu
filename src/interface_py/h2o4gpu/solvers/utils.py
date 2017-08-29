import sys
import numpy as np
from ctypes import *
from h2o4gpu.types import cptr


def device_count(n_gpus=0):
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


# get GPU info, but do in sub-process to avoid mixing parent-child cuda contexts
# https://stackoverflow.com/questions/22950047/cuda-initialization-error-after-fork
def gpu_info():
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(gpu_info_subprocess)
        res = future.result()
    return res


def gpu_info_subprocess():
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
    except Exception as e:
        print("No GPU, setting total_gpus=0")
        print(e)
        sys.stdout.flush()
        pass
    return total_gpus, total_mem


def _to_np(data):
    import pandas as pd
    return data.values if isinstance(data, pd.DataFrame) else data


def _get_data(data, verbose=0):
    # default is no data
    datalocal = None
    m = 0
    n = -1
    fortran = None

    if data is not None:
        try:
            datalocal = _to_np(data)
            fortran = datalocal.flags.f_contiguous
            if datalocal.value is not None:
                # get shapes
                shape_x = np.shape(datalocal)
                m = shape_x[0]
                try:
                    n = shape_x[1]
                except:
                    n = 1
            else:
                if verbose > 0:
                    print('no data')
                n = -1
        except:
            datalocal = _to_np(data)
            # get shapes
            shape_x = np.shape(datalocal)
            m = shape_x[0]
            try:
                n = shape_x[1]
            except:
                n = 1


    else:
        if verbose > 0:
            print('no data')

    return datalocal, m, n, fortran


def _check_data_content(do_check, name, data):
    if do_check == 1:
        assert np.isfinite(data).all(), "%s contains Inf" % name
        assert not np.isnan(data).any(), "%s contains NA" % name


def _check_data_size(data, verbose=0):
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
        except:
            double_precision = -1
        try:
            if data.value is not None:
                m = data.shape[0]
                try:
                    n = data.shape[1]
                except:
                    n = 1
            else:
                m = 0
                n = -1
        except:
            m = data.shape[0]
            try:
                n = data.shape[1]
            except:
                n = 1

    return double_precision, m, n


def _convert_to_ptr(data, c_ftype=c_float):
    null_ptr = POINTER(c_ftype)()

    if data is not None:
        try:
            if data.value is not None:
                data_ptr = cptr(data, dtype=c_ftype)
            else:
                data_ptr = null_ptr
        except:
            data_ptr = cptr(data, dtype=c_ftype)
    else:
        data_ptr = null_ptr

    return data_ptr


def _check_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)
