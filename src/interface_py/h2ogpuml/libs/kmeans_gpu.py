class GPUlib():
    def __init__(self):
        pass

    @staticmethod
    def get():
        from h2ogpuml.libs.utils import gpu_lib_path
        from h2ogpuml.libs.utils import load_kmeans_lib

        return load_kmeans_lib(gpu_lib_path())
