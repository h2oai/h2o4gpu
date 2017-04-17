import os
from ctypes import CDLL, c_int, c_size_t, c_void_p
from pogs.types import c_int_p, c_float_p, c_double_p, c_double_pp, settings_s_p, settings_d_p, solution_s_p, solution_d_p, info_s_p, info_d_p


ext = ".dylib" if os.uname()[0] == "Darwin" else ".so"
lib_path = os.path.join(os.path.dirname(__file__),"../../pypogs_elastic_cpu" + ext)

try:
	if not os.path.exists(lib_path): print("WARNING: Library " + lib_path + " doesn't exist.")
	pogsElasticNetCPU = CDLL(lib_path)
	pogsElasticNetCPU.makePtr.argtypes = [c_int, c_size_t, c_size_t, c_size_t,
										  c_double_p, c_double_p, c_double_p, c_double_p,
										  c_double_pp, c_double_pp, c_double_pp, c_double_pp]
	pogsElasticNetCPU.makePtr.restype = c_int

	print('\nLoaded POGS Elastic Net CPU library.')
except:
	print('\nWarning: POGS Elastic Net CPU shared object (dynamic library) not found at ' + lib_path)
	pogsElasticNetCPU=None


