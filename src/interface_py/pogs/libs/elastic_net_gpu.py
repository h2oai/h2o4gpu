import os
from ctypes import *
from pogs.types import *


ext = ".dylib" if os.uname()[0] == "Darwin" else ".so"
lib_path = os.path.join(os.path.dirname(__file__),"../../cpogs_gpu" + ext)

try:
	if not os.path.exists(lib_path):
		print("WARNING: Library " + lib_path + " doesn't exist.")
	pogsElasticNetGPU = cdll.LoadLibrary(lib_path)
	pogsElasticNetGPU.make_ptr_double.argtypes = [c_int, c_int, c_int, c_size_t, c_size_t, c_size_t, c_double_p, c_double_p, c_double_p, c_double_p, c_void_pp, c_void_pp, c_void_pp, c_void_pp]
	pogsElasticNetGPU.make_ptr_double.restype = c_int

	pogsElasticNetGPU.make_ptr_float.argtypes = [c_int, c_int, c_int, c_size_t, c_size_t, c_size_t, c_float_p, c_float_p, c_float_p, c_float_p, c_void_pp, c_void_pp, c_void_pp, c_void_pp]
	pogsElasticNetGPU.make_ptr_float.restype = c_int
except:
	print('\nWarning: POGS Elastic Net GPU shared object (dynamic library) ' + lib_path + ' failed to load.')
	pogsElasticNetGPU=None


