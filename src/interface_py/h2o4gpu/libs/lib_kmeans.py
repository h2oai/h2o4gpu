"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
from ctypes import c_int, c_size_t, c_float, c_double, cdll
from h2o4gpu.types import c_float_p, c_void_pp, c_double_p


class CPUlib:

    def __init__(self):
        pass

    @staticmethod
    def get():
        from h2o4gpu.libs.lib_utils import cpu_lib_path

        return _load_kmeans_lib(cpu_lib_path())


class GPUlib:

    def __init__(self):
        pass

    @staticmethod
    def get():
        from h2o4gpu.libs.lib_utils import gpu_lib_path

        return _load_kmeans_lib(gpu_lib_path())


def _load_kmeans_lib(lib_path):
    """Load the underlying C/C++ KMeans library using cdll.

    :param lib_path: Path to the library file
    :return: object representing the loaded library
    """
    try:
        h2o4gpu_kmeans_lib = cdll.LoadLibrary(lib_path)

        #Fit and Predict
        h2o4gpu_kmeans_lib.make_ptr_float_kmeans.argtypes = [
            c_int,
            c_int,  # verbose
            c_int,  # seed
            c_int,  # gpu_id
            c_int,  # n_gpus
            c_size_t,  # rows
            c_size_t,  # cols
            c_int,  # data_ord
            c_int,  # n_clusters
            c_int,  # max_iter
            c_int,  # init_from_data
            c_float,  # tol
            c_float_p,  # data
            c_float_p,  # centers
            c_void_pp,  # pred_centers
            c_void_pp  # pred_labels
        ]
        h2o4gpu_kmeans_lib.make_ptr_float_kmeans.restype = c_int

        h2o4gpu_kmeans_lib.make_ptr_double_kmeans.argtypes = [
            c_int,
            c_int,  # verbose
            c_int,  # seed
            c_int,  # gpu_id
            c_int,  # n_gpus
            c_size_t,  # rows
            c_size_t,  # cols
            c_int,  # data_ord
            c_int,  # n_clusters
            c_int,  # max_iter
            c_int,  # init_from_data
            c_double,  # tol
            c_double_p,  # data
            c_double_p,  # centers
            c_void_pp,  # pred_centers
            c_void_pp  # pred_labels
        ]
        h2o4gpu_kmeans_lib.make_ptr_double_kmeans.restype = c_int

        #Transform
        h2o4gpu_kmeans_lib.kmeans_transform_float.argtypes = [
            c_int,  # verbose
            c_int,  # gpu_id
            c_int,  # n_gpus
            c_size_t,  # rows
            c_size_t,  # cols
            c_int,  # data_ord
            c_int,  # k
            c_float_p,  # data
            c_float_p,  # centroids
            c_void_pp
        ]  # result
        h2o4gpu_kmeans_lib.kmeans_transform_float.restype = c_int

        h2o4gpu_kmeans_lib.kmeans_transform_double.argtypes = [
            c_int,  # verbose
            c_int,  # gpu_id
            c_int,  # n_gpus
            c_size_t,  # rows
            c_size_t,  # cols
            c_int,  # data_ord
            c_int,  # k
            c_double_p,  # data
            c_double_p,  # centroids
            c_void_pp
        ]  # result
        h2o4gpu_kmeans_lib.kmeans_transform_double.restype = c_int
    # pylint: disable=broad-except
    except Exception as e:
        print("Exception")
        print(e)
        print(
            '\nWarning: h2o4gpu_kmeans_lib shared object (dynamic library) %s '
            'failed to load. ' % lib_path)
        h2o4gpu_kmeans_lib = None

    return h2o4gpu_kmeans_lib
