from pogs.libs.gpu import pogsGPU
from pogs.solvers.base import BaseSolver

if not pogsGPU:
	print '\nWarning: Cannot create a POGS GPU Solver instance without linking Python module to a compiled POGS GPU libirary'
	print '> Setting pogs.SolverGPU=None'
	print '> Use pogs.SolverCPU(args...) or add CUDA libraries to $PATH and re-run setup.py\n\n'
	SolverGPU=None
else:
	class SolverGPU(object):
		def __init__(self, A, **kwargs):
			self.solver = BaseSolver(A,pogsGPU)
			self.info=self.solver.info
			self.solution=self.solver.pysolution

		def init(self, A, **kwargs):
			self.solver.init(A,**kwargs)
		 
		def solve(self, f, g, **kwargs):
			self.solver.solve(f,g,**kwargs)
		 
		def finish(self):
			self.solver.finish()

		def __delete__(self):
			self.solver.finish()