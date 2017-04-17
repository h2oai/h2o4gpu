import os
from ctypes import CDLL, c_int, c_size_t, c_void_p
from pogs.types import c_int_p, c_float_p, c_double_p, c_double_pp, settings_s_p, settings_d_p, solution_s_p, solution_d_p, info_s_p, info_d_p


ext = ".dylib" if os.uname()[0] == "Darwin" else ".so"
lib_path = os.path.join(os.path.dirname(__file__),"../../pypogs_gpu" + ext)

try:
	pogsElasticNetGPU = CDLL(lib_path)
	pogsElasticNetGPU.makePtr.argtypes = [c_int, c_size_t, c_size_t, c_size_t,
										  c_double_p, c_double_p, c_double_p, c_double_p,
										  c_double_pp, c_double_pp, c_double_pp, c_double_pp]
	pogsElasticNetGPU.makePtr.restype = c_int

	print('\nLoaded POGS Elastic Net GPU library.')
except:
	print('\nWarning: POGS Elastic Net GPU shared object (dynamic library) not found at ' + lib_path)
	pogsElasticNetGPU=None


