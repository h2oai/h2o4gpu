"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
from ctypes import c_int, c_size_t, cdll
from h2o4gpu.types import c_float_p, c_void_pp, c_double_p


class CPUlib:

    def __init__(self):
        pass

    @staticmethod
    def get():
        from h2o4gpu.libs.lib_utils import cpu_lib_path

        return _load_glm_lib(cpu_lib_path())


class GPUlib:

    def __init__(self):
        pass

    @staticmethod
    def get():
        from h2o4gpu.libs.lib_utils import gpu_lib_path

        return _load_glm_lib(gpu_lib_path())


def _load_glm_lib(lib_path):
    """Load the underlying C/C++ GLM library using cdll.

    :param lib_path: Path to the library file
    :return: object representing the loaded library
    """
    try:
        h2o4gpu_glm_lib = cdll.LoadLibrary(lib_path)

        h2o4gpu_glm_lib.make_ptr_double.argtypes = [
            c_int, c_int, c_int, c_size_t, c_size_t, c_size_t, c_int,
            c_double_p, c_double_p, c_double_p, c_double_p, c_double_p,
            c_void_pp, c_void_pp, c_void_pp, c_void_pp, c_void_pp
        ]
        h2o4gpu_glm_lib.make_ptr_double.restype = c_int

        h2o4gpu_glm_lib.make_ptr_float.argtypes = [
            c_int, c_int, c_int, c_size_t, c_size_t, c_size_t, c_int, c_float_p,
            c_float_p, c_float_p, c_float_p, c_float_p, c_void_pp, c_void_pp,
            c_void_pp, c_void_pp, c_void_pp
        ]
        h2o4gpu_glm_lib.make_ptr_float.restype = c_int


# pylint: disable=broad-except
    except Exception as e:
        print("Exception")
        print(e)
        print('\nWarning: h2o4gpu_glm_lib shared object (dynamic library) %s '
              'failed to load. ' % lib_path)
        h2o4gpu_glm_lib = None

    return h2o4gpu_glm_lib
