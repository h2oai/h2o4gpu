"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
class CPUlib:

    def __init__(self):
        pass

    @staticmethod
    def get():
        try:
            import h2o4gpu.libs.ch2o4gpu_cpu as ch2o4gpu_cpu
            return ch2o4gpu_cpu
        except Exception as e:
            print("Exception:")
            print(e)
            print('\nWarning: h2o4gpu shared object (dynamic library) for CPU failed to load.')
            return None

class GPUlib:

    def __init__(self):
        pass

    @staticmethod
    def get():
        try:
            import h2o4gpu.libs.ch2o4gpu_gpu as ch2o4gpu_gpu
            return ch2o4gpu_gpu
        except Exception as e:
            print("Exception:")
            print(e)
            print('\nWarning: h2o4gpu shared object (dynamic library) for GPU failed to load.')
            return None

def getLib(n_gpus, devices):
    cpu_lib = CPUlib().get()
    gpu_lib = GPUlib().get()
    if (n_gpus == 0) or (gpu_lib is None and cpu_lib is not None) or (devices == 0):
        print("\nUsing CPU library\n")
        return cpu_lib
    elif (n_gpus > 0) and (gpu_lib is not None) and (devices > 0):
        print("\nUsing GPU library with %d GPUs\n" % n_gpus)
        return gpu_lib
    else:
        return None
