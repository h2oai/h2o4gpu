import os

def cpu_lib_path():
    return lib_path("ch2ogpuml_cpu")

def gpu_lib_path():
    return lib_path("ch2ogpuml_gpu")

def lib_path(lib_name):
    ext = ".dylib" if os.uname()[0] == "Darwin" else ".so"
    lib_path = os.path.join(os.path.dirname(__file__), "../../" + lib_name + ext)
    if not os.path.exists(lib_path):
        print("WARNING: Library " + lib_path + " doesn't exist.")
    return lib_path