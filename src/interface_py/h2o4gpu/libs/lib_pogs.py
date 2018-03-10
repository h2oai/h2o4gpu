"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
from ctypes import c_int, c_size_t, cdll
from h2o4gpu.types import c_int_p, c_float_p, c_double_p


class CPUlib:

    def __init__(self):
        pass

    @staticmethod
    def get():
        from h2o4gpu.libs.lib_utils import cpu_lib_path

        return _load_pogs_lib(cpu_lib_path())


class GPUlib:

    def __init__(self):
        pass

    @staticmethod
    def get():
        from h2o4gpu.libs.lib_utils import gpu_lib_path

        return _load_pogs_lib(gpu_lib_path())


def _load_pogs_lib(lib_path):
    """Load the underlying C/C++ POGS library using cdll.

    :param lib_path: Path to the library file
    :return: object representing the loaded library
    """
    try:
        h2o4gpu_pogs_lib = cdll.LoadLibrary(lib_path)

        #argument types
        h2o4gpu_pogs_lib.h2o4gpu_init_dense_single.argtypes = [
            c_int, c_int, c_size_t, c_size_t, c_float_p
        ]
        h2o4gpu_pogs_lib.h2o4gpu_init_dense_double.argtypes = [
            c_int, c_int, c_size_t, c_size_t, c_double_p
        ]
        h2o4gpu_pogs_lib.h2o4gpu_init_sparse_single.argtypes = [
            c_int, c_int, c_size_t, c_size_t, c_size_t, c_float_p, c_int_p,
            c_int_p
        ]
        h2o4gpu_pogs_lib.h2o4gpu_init_sparse_double.argtypes = [
            c_int, c_int, c_size_t, c_size_t, c_size_t, c_double_p, c_int_p,
            c_int_p
        ]
        h2o4gpu_pogs_lib.h2o4gpu_solve_single.argtypes = [
            c_void_p, settings_s_p, solution_s_p, info_s_p, c_float_p,
            c_float_p, c_float_p, c_float_p, c_float_p, c_int_p, c_float_p,
            c_float_p, c_float_p, c_float_p, c_float_p, c_int_p
        ]
        h2o4gpu_pogs_lib.h2o4gpu_solve_double.argtypes = [
            c_void_p, settings_d_p, solution_d_p, info_d_p, c_double_p,
            c_double_p, c_double_p, c_double_p, c_double_p, c_int_p, c_double_p,
            c_double_p, c_double_p, c_double_p, c_double_p, c_int_p
        ]
        h2o4gpu_pogs_lib.h2o4gpu_finish_single.argtypes = [c_void_p]
        h2o4gpu_pogs_lib.h2o4gpu_finish_double.argtypes = [c_void_p]

        #return types
        h2o4gpu_pogs_lib.h2o4gpu_init_dense_single.restype = c_void_p
        h2o4gpu_pogs_lib.h2o4gpu_init_dense_double.restype = c_void_p
        h2o4gpu_pogs_lib.h2o4gpu_init_sparse_single.restype = c_void_p
        h2o4gpu_pogs_lib.h2o4gpu_init_sparse_double.restype = c_void_p
        h2o4gpu_pogs_lib.h2o4gpu_solve_single.restype = c_int
        h2o4gpu_pogs_lib.h2o4gpu_solve_double.restype = c_int
        h2o4gpu_pogs_lib.h2o4gpu_finish_single.restype = None
        h2o4gpu_pogs_lib.h2o4gpu_finish_double.restype = None
    # pylint: disable=broad-except
    except Exception as e:
        print("Exception")
        print(e)
        print('\nWarning: h2o4gpu_pogs_lib shared object (dynamic library) %s '
              'failed to load. ' % lib_path)
        h2o4gpu_pogs_lib = None

    return h2o4gpu_pogs_lib
