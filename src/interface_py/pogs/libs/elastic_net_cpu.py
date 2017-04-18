import os
from ctypes import *
from pogs.types import *


ext = ".dylib" if os.uname()[0] == "Darwin" else ".so"
lib_path = os.path.join(os.path.dirname(__file__),"../../cpogs_cpu" + ext)

#try:
print("\n\nTrying to load CPU Library " + lib_path + ".")
if not os.path.exists(lib_path):
    print("WARNING: Library " + lib_path + " doesn't exist.")
pogsElasticNetCPU = cdll.LoadLibrary(lib_path)
print("Loaded CPU Library " + lib_path + ".")
pogsElasticNetCPU.make_ptr_double.argtypes = [c_int, c_size_t, c_size_t, c_size_t,
                                              c_double_p, c_double_p, c_double_p, c_double_p,
                                              c_void_pp, c_void_pp, c_void_pp, c_void_pp]
pogsElasticNetCPU.make_ptr_double.restype = c_int

pogsElasticNetCPU.make_ptr_float.argtypes = [c_int, c_size_t, c_size_t, c_size_t,
                                             c_float_p, c_float_p, c_float_p, c_float_p,
                                             c_void_pp, c_void_pp, c_void_pp, c_void_pp]
pogsElasticNetCPU.make_ptr_float.restype = c_int

print('\nLoaded POGS Elastic Net CPU library.')
#except:
#	print('\nWarning: POGS Elastic Net CPU shared object (dynamic library) not found at ' + lib_path)
#	pogsElasticNetCPU=None


