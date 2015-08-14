from ctypes import *
import numpy as np
import scipy.sparse as sp
import os

#relative library path
uname = os.uname()[0];
if uname == "Darwin":
	ext = ".dylib"
else:
	ext = ".so"


rel_lib_path = "pypogs_gpu" + ext
this_dir = os.path.dirname(__file__)


if this_dir == '':
	pogs = CDLL(rel_lib_path)
else:
	pogs = CDLL(this_dir + "/" + rel_lib_path )


ORD = {}
ORD["COL_MAJ"]=c_int(0)
ORD["ROW_MAJ"]=c_int(1)

FUNCTION = {}
FUNCTION["ABS"] 		=c_int(0)
FUNCTION["EXP"] 		=c_int(1)
FUNCTION["HUBER"] 		=c_int(2)
FUNCTION["IDENTITY"] 	=c_int(3)
FUNCTION["INDBOX01"] 	=c_int(4)
FUNCTION["INDEQ0"]		=c_int(5)
FUNCTION["INDGE0"]		=c_int(6)
FUNCTION["INDLE0"]		=c_int(7)
FUNCTION["LOGISTIC"] 	=c_int(8)
FUNCTION["MAXNEG0"] 	=c_int(9)
FUNCTION["MAXPOS0"] 	=c_int(10)
FUNCTION["NEGENTR"] 	=c_int(11)
FUNCTION["NEGLOG"]		=c_int(12)
FUNCTION["RECIPR"]		=c_int(13)
FUNCTION["SQUARE"]		=c_int(14)
FUNCTION["ZERO"] 		=c_int(15)

STATUS = {}
STATUS[0]='POGS_SUCCESS'
STATUS[1]='POGS_INFEASIBLE'
STATUS[2]='POGS_UNBOUNDED'
STATUS[3]='POGS_MAX_ITER'
STATUS[4]='POGS_NAN_FOUND'
STATUS[5]='POGS_ERROR'

DEFAULTS = {}
DEFAULTS['rho']=1.
DEFAULTS['abs_tol']=1e-4
DEFAULTS['rel_tol']=1e-4
DEFAULTS['max_iters']=2000
DEFAULTS['verbose']=2
DEFAULTS['adaptive_rho']=1
DEFAULTS['gap_stop']=1
DEFAULTS['warm_start']=1


c_int_p = POINTER(c_int)
c_float_p = POINTER(c_float)
c_double_p = POINTER(c_double)


class SettingsS(Structure):
	_fields_ = [('rho', c_float), 
				('abs_tol', c_float), 
				('rel_tol', c_float),
				('max_iters', c_uint), 
				('verbose', c_uint), 
				('adaptive_rho', c_int), 
				('gap_stop', c_int),
				('warm_start', c_int)]

class SettingsD(Structure):
	_fields_ = [('rho', c_double), 
				('abs_tol', c_double), 
				('rel_tol', c_double),
				('max_iters', c_uint), 
				('verbose', c_uint), 
				('adaptive_rho', c_int), 
				('gap_stop', c_int),
				('warm_start', c_int)]

class InfoS(Structure):
	_fields_ = [('iter', c_uint), 
				('status', c_int), 
				('obj',c_float), 
				('rho',c_float)]

class InfoD(Structure):
	_fields_ = [('iter', c_uint), 
				('status', c_int), 
				('obj',c_double), 
				('rho',c_double)]

class Solution(object):
	def __init__(self,double_precision,m,n):
		T = c_double if double_precision else c_float
		self.double_precision = double_precision
		self.x=np.zeros(n,dtype=T)
		self.y=np.zeros(m,dtype=T)
		self.mu=np.zeros(n,dtype=T)
		self.nu=np.zeros(m,dtype=T)

class SolutionS(Structure):
	_fields_ = [('x', c_float_p), 
				('y', c_float_p), 
				('mu', c_float_p), 
				('nu', c_float_p)]

class SolutionD(Structure):
	_fields_ = [('x', c_double_p), 
				('y',c_double_p), 
				('mu',c_double_p), 
				('nu',c_double_p)]


settings_s_p = POINTER(SettingsS)
settings_d_p = POINTER(SettingsD)
info_s_p = POINTER(InfoS)
info_d_p = POINTER(InfoD)
solution_s_p = POINTER(SolutionS)
solution_d_p = POINTER(SolutionD)


#argument types
pogs.pogs_init_dense_single.argtypes = [c_int, c_size_t, c_size_t, c_float_p]
pogs.pogs_init_dense_double.argtypes = [c_int, c_size_t, c_size_t, c_double_p]
pogs.pogs_init_sparse_single.argtypes = [c_int, c_size_t, c_size_t, c_size_t, c_float_p, c_int_p, c_int_p]
pogs.pogs_init_sparse_double.argtypes = [c_int, c_size_t, c_size_t, c_size_t, c_double_p, c_int_p, c_int_p]
pogs.pogs_solve_single.argtypes = [c_void_p, settings_s_p, solution_s_p, info_s_p,
									c_float_p, c_float_p, c_float_p, c_float_p, c_float_p, c_int_p,
									c_float_p, c_float_p, c_float_p, c_float_p, c_float_p, c_int_p]
pogs.pogs_solve_double.argtypes = [c_void_p, settings_d_p, solution_d_p, info_d_p,
									c_double_p, c_double_p, c_double_p, c_double_p, c_double_p, c_int_p,
									c_double_p, c_double_p, c_double_p, c_double_p, c_double_p, c_int_p]
pogs.pogs_finish_single.argtypes = [c_void_p]
pogs.pogs_finish_double.argtypes = [c_void_p]



#return types
pogs.pogs_init_dense_single.restype = c_void_p
pogs.pogs_init_dense_double.restype = c_void_p
pogs.pogs_init_sparse_single.restype = c_void_p
pogs.pogs_init_sparse_double.restype = c_void_p
pogs.pogs_solve_single.restype = c_int
pogs.pogs_solve_double.restype = c_int
pogs.pogs_finish_single.restype = None
pogs.pogs_finish_double.restype = None


def cptr(np_arr,dtype=c_float):
	return np_arr.ctypes.data_as(POINTER(dtype))

def change_settings(settings, **kwargs):
	if 'rho' in kwargs: settings.rho=kwargs['rho']
	if 'abs_tol' in kwargs: settings.abs_tol=kwargs['abs_tol']
	if 'rel_tol' in kwargs: settings.rel_tol=kwargs['rel_tol']
	if 'max_iters' in kwargs: settings.max_iters=kwargs['max_iters']
	if 'verbose' in kwargs: settings.verbose=kwargs['verbose']
	if 'adaptive_rho' in kwargs: settings.adaptive_rho=kwargs['adaptive_rho']
	if 'gap_stop' in kwargs: settings.gap_stop=kwargs['gap_stop']
	if 'warm_start' in kwargs: settings.warm_start=kwargs['warm_start']

def make_settings(double_precision=False, **kwargs):
	rho = kwargs['rho'] if 'rho' in kwargs.keys() else DEFAULTS['rho'] 
	relt = kwargs['abs_tol'] if 'abs_tol' in kwargs.keys() else DEFAULTS['abs_tol'] 
	abst = kwargs['rel_tol'] if 'rel_tol' in kwargs.keys() else DEFAULTS['rel_tol'] 
	maxit = kwargs['max_iters'] if 'max_iters' in kwargs.keys() else DEFAULTS['max_iters'] 
	verb = kwargs['verbose'] if 'verbose' in kwargs.keys() else DEFAULTS['verbose'] 
	adap = kwargs['adaptive_rho'] if 'adaptive_rho' in kwargs.keys() else DEFAULTS['adaptive_rho'] 
	gaps = kwargs['gap_stop'] if 'gap_stop' in kwargs.keys() else DEFAULTS['gap_stop']
	warm = kwargs['warm_start'] if 'warm_start' in kwargs.keys() else DEFAULTS['warm_start']
	if double_precision:
		return SettingsD(rho, relt, abst, maxit, verb, adap, gaps, warm)
	else:
		return SettingsS(rho, relt, abst, maxit, verb, adap, gaps, warm)

def change_solution(solution, **kwargs):
	if 'x_init' in kwargs: solution.x = x_init
	if 'nu_init' in kwargs: solution.nu = nu_init

def make_solution(pysol):
	if pysol.double_precision:
		return SolutionD(cptr(pysol.x,c_double),cptr(pysol.y,c_double),
							cptr(pysol.mu,c_double),cptr(pysol.nu,c_double))
	else:
		return SolutionS(cptr(pysol.x,c_float),cptr(pysol.y,c_float),
		 					cptr(pysol.mu,c_float),cptr(pysol.nu,c_float))

		
def make_info(double_precision):
	if double_precision:
		return InfoD(0,0,np.inf,0)
	else:
		return InfoS(0,0,np.inf,0)


class FunctionVector(object):
	def __init__(self, length, double_precision=False):
		T = c_double if double_precision else c_float
		self.a = np.ones(length,T)
		self.b = np.zeros(length,T)
		self.c = np.ones(length,T)
		self.d = np.zeros(length,T)
		self.e = np.zeros(length,T)
		self.h = np.zeros(length, c_int)

class Solver(object):
	def __init__(self, m, n, A, **kwargs):
		try:
			self.m=m
			self.n=n
			self.A=A
			self.dense = isinstance(A,np.ndarray)
			self.CSC = isinstance(A, sp.csc.csc_matrix)
			self.CSR = isinstance(A, sp.csr.csr_matrix)
			# 
			assert self.dense or self.CSC or self.CSR
			assert A.dtype == c_float or A.dtype == c_double
			# 
			self.double_precision = A.dtype == c_double
			self.settings = make_settings(self.double_precision, **kwargs)
			self.pysolution = Solution(self.double_precision,m,n)
			self.solution = make_solution(self.pysolution)
			self.info = make_info(self.double_precision)
			self.order = ORD["ROW_MAJ"] if (self.CSR or self.dense) else ORD["COL_MAJ"]
			# 
			if self.dense and not self.double_precision:
				self.work = pogs.pogs_init_dense_single(self.order, m, n, cptr(A,c_float))
			elif self.dense:
				self.work = pogs.pogs_init_dense_double(self.order, m, n, cptr(A,c_double))
			elif not self.double_precision:
				self.work = pogs.pogs_init_sparse_single(self.order, m, n, A.nnz, cptr(A.data,c_float), 
														 cptr(A.indices,c_int), cptr(A.indptr,c_int))
			else:
				self.work = pogs.pogs_init_sparse_double(self.order, m, n, A.nnz, cptr(A.data,c_double),
														 cptr(A.indices,c_int), cptr(A.indptr,c_int))
		# 
		except AssertionError:
			print "data must be a numpy ndarray or scipy csc_matrix containing float32 or float64 values"
	# 
	def init(self, m, n, A, **kwargs):
		if not self.work:
			self.__init__(m,n,A, **kwargs)
		else:
			print "POGS_work already intialized, cannot re-initialize without calling finish()"
	# 
	def solve(self, f, g, **kwargs):
		try:
			#assert f,g types
			assert isinstance(f, FunctionVector)
			assert isinstance(g, FunctionVector)
			# 
			change_settings(self.settings, **kwargs)
			change_solution(self.solution, **kwargs)
			# 
			if not self.work:
				print "no viable POGS_work pointer to call solve(). call Solver.init( args... )"
				return 
			elif not self.double_precision:
				pogs.pogs_solve_single(self.work, pointer(self.settings), pointer(self.solution), pointer(self.info),
										cptr(f.a,c_float), cptr(f.b,c_float), cptr(f.c,c_float), 
										cptr(f.d,c_float), cptr(f.e,c_float), cptr(f.h,c_int),
										cptr(g.a,c_float), cptr(g.b,c_float), cptr(g.c,c_float),
										cptr(g.d,c_float), cptr(g.e,c_float), cptr(g.h,c_int))
			else:
				pogs.pogs_solve_double(self.work, pointer(self.settings), pointer(self.solution), pointer(self.info), 
										cptr(f.a,c_double), cptr(f.b,c_double), cptr(f.c,c_double), 
										cptr(f.d,c_double), cptr(f.e,c_double), cptr(f.h,c_int),
										cptr(g.a,c_double), cptr(g.b,c_double), cptr(g.c,c_double),
										cptr(g.d,c_double), cptr(g.e,c_double), cptr(g.h,c_int))
			# 
		except AssertionError:
			print "f and g must be of type FunctionVector"
	# 
	def finish(self):
		if not self.work:
			print "no viable POGS_work pointer to call finish(). call Solver.init( args... )"
			pass
		elif not self.double_precision:
			pogs.pogs_finish_single(self.work)
			self.work = None
		else:
			pogs.pogs_finish_double(self.work)
			self.work = None
		print "shutting down... POGS_work freed in C++"

	def __delete__(self):
		self.finish()








