"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import os


def cpu_lib_path():
    return get_lib_path("ch2o4gpu_cpu")


def gpu_lib_path():
    return get_lib_path("ch2o4gpu_gpu")


def get_lib_path(lib_name):
    ext = ".dylib" if os.uname()[0] == "Darwin" else ".so"
    lib_path = os.path.join(
        os.path.dirname(__file__), "../../" + lib_name + ext)
    if not os.path.exists(lib_path):
        print("WARNING: Library " + lib_path + " doesn't exist.")
    return lib_path
