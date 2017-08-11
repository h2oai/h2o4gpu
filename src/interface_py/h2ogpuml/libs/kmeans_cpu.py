class CPUlib():
    def __init__(self):
        pass

    @staticmethod
    def get():
        from h2ogpuml.libs.utils import cpu_lib_path
        from h2ogpuml.libs.utils import load_kmeans_lib

        return load_kmeans_lib(cpu_lib_path())
