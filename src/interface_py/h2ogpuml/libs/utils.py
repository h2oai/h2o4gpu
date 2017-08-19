from ctypes import *
from h2ogpuml.types import *
import os


def cpu_lib_path():
    return get_lib_path("ch2ogpuml_cpu")


def gpu_lib_path():
    return get_lib_path("ch2ogpuml_gpu")


def get_lib_path(lib_name):
    ext = ".dylib" if os.uname()[0] == "Darwin" else ".so"
    lib_path = os.path.join(os.path.dirname(__file__),
                            "../../" + lib_name + ext)
    if not os.path.exists(lib_path):
        print("WARNING: Library " + lib_path + " doesn't exist.")
    return lib_path


def load_kmeans_lib(lib_path):
    try:
        h2ogpuml_kmeans_lib = cdll.LoadLibrary(lib_path)

        # Fit and Predict
        h2ogpuml_kmeans_lib.make_ptr_float_kmeans.argtypes = [c_int, c_int,
                                                              c_int, c_int,
                                                              c_int,
                                                              c_size_t,
                                                              c_size_t,
                                                              c_int,
                                                              c_int, c_int,
                                                              c_int,
                                                              c_int, c_int,
                                                              c_float,
                                                              c_float_p,
                                                              c_int_p,
                                                              c_float_p,
                                                              c_void_pp,
                                                              c_void_pp]
        h2ogpuml_kmeans_lib.make_ptr_float_kmeans.restype = c_int

        h2ogpuml_kmeans_lib.make_ptr_double_kmeans.argtypes = [c_int, c_int,
                                                               c_int, c_int,
                                                               c_int,
                                                               c_size_t,
                                                               c_size_t,
                                                               c_int, c_int,
                                                               c_int, c_int,
                                                               c_int, c_int,
                                                               c_double,
                                                               c_double_p,
                                                               c_int_p,
                                                               c_double_p,
                                                               c_void_pp,
                                                               c_void_pp]
        h2ogpuml_kmeans_lib.make_ptr_double_kmeans.restype = c_int

        # Transform
        h2ogpuml_kmeans_lib.kmeans_transform_float.argtypes = [
            c_int,  # verbose
            c_int,  # gpu_id
            c_int,  # n_gpus
            c_size_t,  # rows
            c_size_t,  # cols
            c_int,  # data_ord
            c_int, # k
            c_float_p,  # data
            c_float_p,  # centroids
            c_void_pp]  # result
        h2ogpuml_kmeans_lib.kmeans_transform_float.restype = c_int

        h2ogpuml_kmeans_lib.kmeans_transform_double.argtypes = [
            c_int,  # verbose
            c_int,  # gpu_id
            c_int,  # n_gpus
            c_size_t,  # rows
            c_size_t,  # cols
            c_int,  # data_ord
            c_int, # k
            c_double_p,  # data
            c_double_p,  # centroids
            c_void_pp]  # result
        h2ogpuml_kmeans_lib.kmeans_transform_double.restype = c_int

    except:
        print(
            '\nWarning: h2ogpuml_kmeans_lib shared object (dynamic library) %s failed to load.'
            % lib_path)
        h2ogpuml_kmeans_lib = None
    return h2ogpuml_kmeans_lib