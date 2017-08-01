from ctypes import *
from h2ogpuml.types import *


class GPUlib():
    def __init__(self):
        pass

    def get(self):
        from h2ogpuml.libs.utils import gpu_lib_path
        lib_path = gpu_lib_path()

        try:
            h2ogpumlKMeansGPU = cdll.LoadLibrary(lib_path)

            h2ogpumlKMeansGPU.make_ptr_float_kmeans.argtypes = [c_int, c_int, c_int, c_int, c_size_t, c_size_t, c_int, c_int, c_int, c_int,
                                                                c_int, c_int, c_float, c_float_p, c_int_p, c_void_pp]
            h2ogpumlKMeansGPU.make_ptr_float_kmeans.restype = c_int

            h2ogpumlKMeansGPU.make_ptr_double_kmeans.argtypes = [c_int, c_int, c_int, c_int, c_size_t, c_size_t, c_int, c_int, c_int, c_int,
                                                                 c_int, c_int, c_double, c_double_p, c_int_p, c_void_pp]
            h2ogpumlKMeansGPU.make_ptr_double_kmeans.restype = c_int
        except:
            print('\nWarning: h2ogpumlKMeansGPU shared object (dynamic library) ' + lib_path + ' failed to load.')
            h2ogpumlKMeansGPU = None
        return h2ogpumlKMeansGPU