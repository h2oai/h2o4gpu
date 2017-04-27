import os, sys, traceback, logging
from ctypes import CDLL, c_int, c_size_t, c_void_p
from pogs.types import c_int_p, c_float_p, c_double_p, settings_s_p, settings_d_p, solution_s_p, solution_d_p, info_s_p, info_d_p


ext = ".dylib" if os.uname()[0] == "Darwin" else ".so"
lib_path = os.path.join(os.path.dirname(__file__),"../../cpogs_gpu" + ext)


try:
      pogsGPU = CDLL(lib_path)

      #argument types
      pogsGPU.pogs_init_dense_single.argtypes = [c_int, c_int, c_size_t, c_size_t, c_float_p]
      pogsGPU.pogs_init_dense_double.argtypes = [c_int, c_int, c_size_t, c_size_t, c_double_p]
      pogsGPU.pogs_init_sparse_single.argtypes = [c_int, c_int, c_size_t, c_size_t, c_size_t, c_float_p, c_int_p, c_int_p]
      pogsGPU.pogs_init_sparse_double.argtypes = [c_int, c_int, c_size_t, c_size_t, c_size_t, c_double_p, c_int_p, c_int_p]
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

      print('\nLoaded POGS GPU library.')
except:
      logging.exception("in gpu.py")
      print('\nWarning: POGS GPU shared object (dynamic library) not found at ' + lib_path)
      pogsGPU=None


