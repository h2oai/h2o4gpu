"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import ctypes


class parameters(ctypes.Structure):
    _fields_ = [('X_n', ctypes.c_int), ('X_m', ctypes.c_int),
                ('k', ctypes.c_int), ('algorithm', ctypes.c_char_p),
                ('n_iter', ctypes.c_int), ('random_state', ctypes.c_int),
                ('tol', ctypes.c_float),
                ('verbose', ctypes.c_int),
                ('gpu_id', ctypes.c_int)]


class CPUlib:

    def __init__(self):
        pass

    @staticmethod
    def get():
        from h2o4gpu.libs.lib_utils import cpu_lib_path

        return _load_tsvd_lib(cpu_lib_path())


class GPUlib:

    def __init__(self):
        pass

    @staticmethod
    def get():
        from h2o4gpu.libs.lib_utils import gpu_lib_path

        return _load_tsvd_lib(gpu_lib_path())


def _load_tsvd_lib(lib_path):
    """Load the underlying C/C++ tsvd library using cdll.

    :param lib_path: Path to the library file
    :return: object representing the loaded library
    """
    try:
        h2o4gpu_tsvd_lib = ctypes.cdll.LoadLibrary(lib_path)

        #Original Version
        h2o4gpu_tsvd_lib.truncated_svd.argtypes = \
            [ctypes.POINTER(ctypes.c_double),
             ctypes.POINTER(ctypes.c_double),
             ctypes.POINTER(ctypes.c_double),
             ctypes.POINTER(ctypes.c_double),
             parameters]

        #Float version
        h2o4gpu_tsvd_lib.truncated_svd_float.argtypes = \
            [ctypes.POINTER(ctypes.c_float),
             ctypes.POINTER(ctypes.c_float),
             ctypes.POINTER(ctypes.c_float),
             ctypes.POINTER(ctypes.c_float),
             parameters]

        #Double version
        h2o4gpu_tsvd_lib.truncated_svd_double.argtypes = \
            [ctypes.POINTER(ctypes.c_double),
             ctypes.POINTER(ctypes.c_double),
             ctypes.POINTER(ctypes.c_double),
             ctypes.POINTER(ctypes.c_double),
             parameters]

    # pylint: disable=broad-except
    except Exception as e:
        print("Exception")
        print(e)
        print('\nWarning: h2o4gpu_tsvd_lib shared object (dynamic library) %s '
              'failed to load. ' % lib_path)
        h2o4gpu_tsvd_lib = None

    return h2o4gpu_tsvd_lib
