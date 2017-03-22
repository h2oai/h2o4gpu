from ctypes import POINTER, c_int, c_uint, c_float, c_double, Structure
from numpy import float32, float64, zeros, ones, inf

# POGS constants
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

# Default POGS solver settings
DEFAULTS = {}
DEFAULTS['rho']			=	1. 		# rho = 1.0
DEFAULTS['abs_tol']		=	1e-4 	# abs_tol = 1e-2
DEFAULTS['rel_tol']		=	1e-4 	# rel_tol = 1e-4
DEFAULTS['max_iters']	=	2500 	# max_iters = 2500
DEFAULTS['verbose']		=	2 		# verbose = 2
DEFAULTS['adaptive_rho']=	1 		# adaptive_rho = True
DEFAULTS['gap_stop']	=	1		# gap_stop = True
DEFAULTS['warm_start']	=	0 		# warm_start = False

# pointers to C types
c_int_p = POINTER(c_int)
c_float_p = POINTER(c_float)
c_double_p = POINTER(c_double)

# POGS types
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
				('rho',c_float),
				('solvetime',c_float)]

class InfoD(Structure):
	_fields_ = [('iter', c_uint), 
				('status', c_int), 
				('obj',c_double), 
				('rho',c_double),
				('solvetime',c_float)]


class Solution(object):
	def __init__(self,double_precision,m,n):
		T = c_double if double_precision else c_float
		self.double_precision = double_precision
		self.x=zeros(n,dtype=T)
		self.y=zeros(m,dtype=T)
		self.mu=zeros(n,dtype=T)
		self.nu=zeros(m,dtype=T)

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

# pointers to POGS types
settings_s_p = POINTER(SettingsS)
settings_d_p = POINTER(SettingsD)
info_s_p = POINTER(InfoS)
info_d_p = POINTER(InfoD)
solution_s_p = POINTER(SolutionS)
solution_d_p = POINTER(SolutionD)

def cptr(np_arr,dtype=c_float):
	return np_arr.ctypes.data_as(POINTER(dtype))

def change_settings(settings, **kwargs):
	
	# all settings (except warm_start) are persistent and change only if called
	if 'rho' in kwargs: settings.rho=kwargs['rho']
	if 'abs_tol' in kwargs: settings.abs_tol=kwargs['abs_tol']
	if 'rel_tol' in kwargs: settings.rel_tol=kwargs['rel_tol']
	if 'max_iters' in kwargs: settings.max_iters=kwargs['max_iters']
	if 'verbose' in kwargs: settings.verbose=kwargs['verbose']
	if 'adaptive_rho' in kwargs: settings.adaptive_rho=kwargs['adaptive_rho']
	if 'gap_stop' in kwargs: settings.gap_stop=kwargs['gap_stop']
	
	# warm_start must be specified each time it is desired
	if 'warm_start' in kwargs: 
		settings.warm_start=kwargs['warm_start']
	else:
		settings.warm_start=0


def make_settings(double_precision=False, **kwargs):
	rho = kwargs['rho'] if 'rho' in list(kwargs.keys()) else DEFAULTS['rho'] 
	relt = kwargs['abs_tol'] if 'abs_tol' in list(kwargs.keys()) else DEFAULTS['abs_tol'] 
	abst = kwargs['rel_tol'] if 'rel_tol' in list(kwargs.keys()) else DEFAULTS['rel_tol'] 
	maxit = kwargs['max_iters'] if 'max_iters' in list(kwargs.keys()) else DEFAULTS['max_iters'] 
	verb = kwargs['verbose'] if 'verbose' in list(kwargs.keys()) else DEFAULTS['verbose'] 
	adap = kwargs['adaptive_rho'] if 'adaptive_rho' in list(kwargs.keys()) else DEFAULTS['adaptive_rho'] 
	gaps = kwargs['gap_stop'] if 'gap_stop' in list(kwargs.keys()) else DEFAULTS['gap_stop']
	warm = kwargs['warm_start'] if 'warm_start' in list(kwargs.keys()) else DEFAULTS['warm_start']
	if double_precision:
		return SettingsD(rho, relt, abst, maxit, verb, adap, gaps, warm)
	else:
		return SettingsS(rho, relt, abst, maxit, verb, adap, gaps, warm)

def change_solution(pysolution, **kwargs):
	try:
		if 'x_init' in kwargs: pysolution.x[:] = kwargs['x_init'][:]
		if 'nu_init' in kwargs: pysolution.nu[:] = kwargs['nu_init'][:]
	except:
		#TODO: message about vector lengths? 
		raise

def make_solution(pysolution):
	if pysolution.double_precision:
		return SolutionD(cptr(pysolution.x,c_double),cptr(pysolution.y,c_double),
							cptr(pysolution.mu,c_double),cptr(pysolution.nu,c_double))
	else:
		return SolutionS(cptr(pysolution.x,c_float),cptr(pysolution.y,c_float),
		 					cptr(pysolution.mu,c_float),cptr(pysolution.nu,c_float))

		
def make_info(double_precision):
	if double_precision:
		return InfoD(0,0,inf,0,0)
	else:
		return InfoS(0,0,inf,0,0)


class FunctionVector(object):
	def __init__(self, length, double_precision=False):
		T = c_double if double_precision else c_float
		self.a = ones(length,T)
		self.b = zeros(length,T)
		self.c = ones(length,T)
		self.d = zeros(length,T)
		self.e = zeros(length,T)
		self.h = zeros(length, c_int)
		self.double_precision = double_precision

	def length(self):
		return len(self.a)

	def copyfrom(self,f):
		self.a[:]=f.a[:]
		self.b[:]=f.b[:]
		self.c[:]=f.c[:]
		self.d[:]=f.d[:]
		self.e[:]=f.e[:]
		self.h[:]=f.h[:]

	def copyto(self,f):
		f.a[:]=self.a[:]
		f.b[:]=self.b[:]
		f.c[:]=self.c[:]
		f.d[:]=self.d[:]
		f.e[:]=self.e[:]
		f.h[:]=self.h[:]


	def to_double(self):
		if self.double_precision:
			return self
		else:
			f=FunctionVector(self.length(),double_precision=True)
			self.copyto(f)
			return f

	def to_float(self):
		if self.double_precision:
			f=FunctionVector(self.length())
			self.copyto(f)
			return f 
		else:
			return self


