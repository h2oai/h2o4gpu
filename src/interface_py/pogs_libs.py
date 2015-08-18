from ctypes import *
import os
from pogs_types import *

#relative library path
uname = os.uname()[0];
if uname == "Darwin":
	ext = ".dylib"
else:
	ext = ".so"

this_dir = os.path.dirname(__file__)
rel_lib_path = "../interface_c/libpogs" + ext

try:
	if this_dir == '':
		pogsCPU = CDLL(rel_lib_path)
	else:
		pogsCPU = CDLL(this_dir + "/" + rel_lib_path )

	#argument types
	pogsCPU.pogs_init_dense_single.argtypes = [c_int, c_size_t, c_size_t, c_float_p]
	pogsCPU.pogs_init_dense_double.argtypes = [c_int, c_size_t, c_size_t, c_double_p]
	pogsCPU.pogs_init_sparse_single.argtypes = [c_int, c_size_t, c_size_t, c_size_t, c_float_p, c_int_p, c_int_p]
	pogsCPU.pogs_init_sparse_double.argtypes = [c_int, c_size_t, c_size_t, c_size_t, c_double_p, c_int_p, c_int_p]
	pogsCPU.pogs_solve_single.argtypes = [c_void_p, settings_s_p, solution_s_p, info_s_p,
										c_float_p, c_float_p, c_float_p, c_float_p, c_float_p, c_int_p,
										c_float_p, c_float_p, c_float_p, c_float_p, c_float_p, c_int_p]
	pogsCPU.pogs_solve_double.argtypes = [c_void_p, settings_d_p, solution_d_p, info_d_p,
										c_double_p, c_double_p, c_double_p, c_double_p, c_double_p, c_int_p,
										c_double_p, c_double_p, c_double_p, c_double_p, c_double_p, c_int_p]
	pogsCPU.pogs_finish_single.argtypes = [c_void_p]
	pogsCPU.pogs_finish_double.argtypes = [c_void_p]



	#return types
	pogsCPU.pogs_init_dense_single.restype = c_void_p
	pogsCPU.pogs_init_dense_double.restype = c_void_p
	pogsCPU.pogs_init_sparse_single.restype = c_void_p
	pogsCPU.pogs_init_sparse_double.restype = c_void_p
	pogsCPU.pogs_solve_single.restype = c_int
	pogsCPU.pogs_solve_double.restype = c_int
	pogsCPU.pogs_finish_single.restype = None
	pogsCPU.pogs_finish_double.restype = None

except:
	'POGS CPU shared object (dynamic library) not found'
	pogCPU=None




rel_lib_path = "../interface_c/pypogs_gpu" + ext
try:
	if this_dir == '':
		pogsGPU = CDLL(rel_lib_path)
	else:
		pogsGPU = CDLL(this_dir + "/" + rel_lib_path )


	#argument types
	pogsGPU.pogs_init_dense_single.argtypes = [c_int, c_size_t, c_size_t, c_float_p]
	pogsGPU.pogs_init_dense_double.argtypes = [c_int, c_size_t, c_size_t, c_double_p]
	pogsGPU.pogs_init_sparse_single.argtypes = [c_int, c_size_t, c_size_t, c_size_t, c_float_p, c_int_p, c_int_p]
	pogsGPU.pogs_init_sparse_double.argtypes = [c_int, c_size_t, c_size_t, c_size_t, c_double_p, c_int_p, c_int_p]
	pogsGPU.pogs_solve_single.argtypes = [c_void_p, settings_s_p, solution_s_p, info_s_p,
										c_float_p, c_float_p, c_float_p, c_float_p, c_float_p, c_int_p,
										c_float_p, c_float_p, c_float_p, c_float_p, c_float_p, c_int_p]
	pogsGPU.pogs_solve_double.argtypes = [c_void_p, settings_d_p, solution_d_p, info_d_p,
										c_double_p, c_double_p, c_double_p, c_double_p, c_double_p, c_int_p,
										c_double_p, c_double_p, c_double_p, c_double_p, c_double_p, c_int_p]
	pogsGPU.pogs_finish_single.argtypes = [c_void_p]
	pogsGPU.pogs_finish_double.argtypes = [c_void_p]



	#return types
	pogsGPU.pogs_init_dense_single.restype = c_void_p
	pogsGPU.pogs_init_dense_double.restype = c_void_p
	pogsGPU.pogs_init_sparse_single.restype = c_void_p
	pogsGPU.pogs_init_sparse_double.restype = c_void_p
	pogsGPU.pogs_solve_single.restype = c_int
	pogsGPU.pogs_solve_double.restype = c_int
	pogsGPU.pogs_finish_single.restype = None
	pogsGPU.pogs_finish_double.restype = None


except:
	'POGS GPU shared object (dynamic library) not found'
	pogsGPU=None


