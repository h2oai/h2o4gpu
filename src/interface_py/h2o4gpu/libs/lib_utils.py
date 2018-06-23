"""
:copyright: 2017-2018 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import warnings
# pylint: disable=unused-variable
class CPUlib(object):
    """H2O4GPU CPU module"""

    def __init__(self):
        pass

    @staticmethod
    def get(verbose=0):
        """Get the CPU module object"""
        # SWIG generated files contain some deprecated calls to imp
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import h2o4gpu.libs.ch2o4gpu_cpu as ch2o4gpu_cpu
                return ch2o4gpu_cpu
            except ImportError as e:
                if verbose > 0:
                    print("Exception:")
                    print(e)
                    print('\nWarning: h2o4gpu shared object (dynamic library)'
                          ' for CPU failed to load.')
                return None


# pylint: disable=unused-variable
class GPUlib(object):
    """H2O4GPU GPU module"""

    def __init__(self):
        pass

    @staticmethod
    def get(verbose=0):
        """Get the GPU module object"""
        # SWIG generated files contain some deprecated calls to imp
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import h2o4gpu.libs.ch2o4gpu_gpu as ch2o4gpu_gpu
                return ch2o4gpu_gpu
            except ImportError as e:
                if verbose > 0:
                    print("Exception:")
                    print(e)
                    print('\nWarning: h2o4gpu shared object (dynamic library)'
                          ' for GPU failed to load.')
                return None


def get_lib(n_gpus, devices, verbose=0):
    """Load either CPU or GPU H2O4GPU library."""
    cpu_lib = CPUlib().get(verbose=verbose)
    gpu_lib = GPUlib().get(verbose=verbose)
    if (n_gpus == 0) or \
            (gpu_lib is None and cpu_lib is not None) or \
            (devices == 0):
        if verbose > 0:
            print("\nUsing CPU library\n")
        return cpu_lib
    elif (n_gpus > 0) and (gpu_lib is not None) and (devices > 0):
        if verbose > 0:
            print("\nUsing GPU library with %d GPUs\n" % n_gpus)
        return gpu_lib
    else:
        return None
