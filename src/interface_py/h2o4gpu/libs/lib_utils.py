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