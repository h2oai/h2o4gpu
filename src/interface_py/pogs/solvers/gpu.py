# from pogs.libs.gpu import pogsGPU
# from pogs.solvers import BaseSolver

# if not pogsGPU:
# 	print '\nWarning: Cannot create a POGS GPU Solver instance without linking Python module to a compiled POGS GPU libirary'
# 	print '> Setting pogs.SolverGPU=None'
# 	print '> Use pogs.SolverCPU(args...) or add CUDA libraries to $PATH and re-run setup.py\n\n'
# 	SolverGPU=None
# else:
# 	class SolverGPU(object):
# 		def __init__(self, A, **kwargs):
# 			self.solver = BaseSolver(A,pogsGPU)

# 		def init(self, A, **kwargs):
# 			self.solver.init(A,**kwargs)
		 
# 		def solve(self, f, g, **kwargs):
# 			self.solver.solve(f,g,**kwargs)
		 
# 		def finish(self):
# 			self.solver.finish()

# 		def __delete__(self):
# 			self.solver.finish()

from ctypes import c_int, c_float, c_double, pointer
from numpy import ndarray
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from pogs.types import ORD, cptr, make_settings, make_solution, make_info, change_settings, change_solution, Solution, FunctionVector
from pogs.libs.gpu import pogsGPU

#TODO: catch Ctrl-C

if not pogsGPU:
	print '\nWarning: Cannot create a POGS GPU Solver instance without linking Python module to a compiled POGS GPU libirary'
	print '> Setting pogs.SolverGPU=None'
	print '> Use pogs.SolverCPU(args...) or add CUDA libraries to $PATH and re-run setup.py\n\n'
	SolverGPU=None
else:
	class SolverGPU(object):
		def __init__(self, A, **kwargs):
			try:
				self.dense = isinstance(A,ndarray) and len(A.shape)==2
				self.CSC = isinstance(A, csc_matrix)
				self.CSR = isinstance(A, csr_matrix)

				assert self.dense or self.CSC or self.CSR
				assert A.dtype == c_float or A.dtype == c_double
				
				self.m = A.shape[0]
				self.n = A.shape[1]
				self.A=A	
				
				self.double_precision = A.dtype == c_double
				self.settings = make_settings(self.double_precision, **kwargs)
				self.pysolution = Solution(self.double_precision,self.m,self.n)
				self.solution = make_solution(self.pysolution)
				self.info = make_info(self.double_precision)
				self.order = ORD["ROW_MAJ"] if (self.CSR or self.dense) else ORD["COL_MAJ"]
				 
				if self.dense and not self.double_precision:
					self.work = pogsGPU.pogs_init_dense_single(self.order, self.m, self.n, cptr(A,c_float))
				elif self.dense:
					self.work = pogsGPU.pogs_init_dense_double(self.order, self.m, self.n, cptr(A,c_double))
				elif not self.double_precision:
					self.work = pogsGPU.pogs_init_sparse_single(self.order, self.m, self.n, A.nnz, cptr(A.data,c_float), 
															 cptr(A.indices,c_int), cptr(A.indptr,c_int))
				else:
					self.work = pogsGPU.pogs_init_sparse_double(self.order, self.m, self.n, A.nnz, cptr(A.data,c_double),
															 cptr(A.indices,c_int), cptr(A.indptr,c_int))
			
			except AssertionError:
				print "data must be a numpy ndarray or scipy csc_matrix containing float32 or float64 values"


		def init(self, A, **kwargs):
			if not self.work:
				self.__init__(A, **kwargs)
			else:
				print "POGS_work already intialized, cannot re-initialize without calling finish()"
		 
		def solve(self, f, g, **kwargs):
			try:
				#assert f,g types
				assert isinstance(f, FunctionVector)
				assert isinstance(g, FunctionVector)
				

				# pass previous rho through, if relevant
				if self.info.rho>0:
					self.settings.rho=self.info.rho

				# apply user inputs
				change_settings(self.settings, **kwargs)
				change_solution(self.solution, **kwargs)
				 
				if not self.work:
					print "no viable POGS_work pointer to call solve(). call Solver.init( args... )"
					return 
				elif not self.double_precision:
					pogsGPU.pogs_solve_single(self.work, pointer(self.settings), pointer(self.solution), pointer(self.info),
											cptr(f.a,c_float), cptr(f.b,c_float), cptr(f.c,c_float), 
											cptr(f.d,c_float), cptr(f.e,c_float), cptr(f.h,c_int),
											cptr(g.a,c_float), cptr(g.b,c_float), cptr(g.c,c_float),
											cptr(g.d,c_float), cptr(g.e,c_float), cptr(g.h,c_int))
				else:
					pogsGPU.pogs_solve_double(self.work, pointer(self.settings), pointer(self.solution), pointer(self.info), 
											cptr(f.a,c_double), cptr(f.b,c_double), cptr(f.c,c_double), 
											cptr(f.d,c_double), cptr(f.e,c_double), cptr(f.h,c_int),
											cptr(g.a,c_double), cptr(g.b,c_double), cptr(g.c,c_double),
											cptr(g.d,c_double), cptr(g.e,c_double), cptr(g.h,c_int))
				 
			except AssertionError:
				print "f and g must be of type FunctionVector"
		 
		def finish(self):
			if not self.work:
				print "no viable POGS_work pointer to call finish(). call Solver.init( args... )"
				pass
			elif not self.double_precision:
				pogsGPU.pogs_finish_single(self.work)
				self.work = None
			else:
				pogsGPU.pogs_finish_double(self.work)
				self.work = None
			print "shutting down... POGS_work freed in C++"

		def __delete__(self):
			self.finish()