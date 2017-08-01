from ctypes import *
from h2ogpuml.types import *


class CPUlib():
    def __init__(self):
        pass

    def get(self):
        from h2ogpuml.libs.utils import cpu_lib_path
        lib_path = cpu_lib_path()

        try:
            h2ogpumlKMeansCPU = cdll.LoadLibrary(lib_path)

            h2ogpumlKMeansCPU.make_ptr_float_kmeans.argtypes = [c_int, c_int, c_int, c_int, c_size_t, c_size_t, c_int, c_int, c_int, c_int,
                                                                c_int, c_int, c_float, c_float_p, c_int_p, c_void_pp]
            h2ogpumlKMeansCPU.make_ptr_float_kmeans.restype = c_int

            h2ogpumlKMeansCPU.make_ptr_double_kmeans.argtypes = [c_int, c_int, c_int, c_int, c_size_t, c_size_t, c_int, c_int, c_int, c_int,
                                                                 c_int, c_int, c_double, c_double_p, c_int_p, c_void_pp]
            h2ogpumlKMeansCPU.make_ptr_double_kmeans.restype = c_int
        except:
            print('\nWarning: h2ogpumlKMeansCPU shared object (dynamic library) ' + lib_path + ' failed to load.')
            h2ogpumlKMeansCPU = None
        return h2ogpumlKMeansCPU
