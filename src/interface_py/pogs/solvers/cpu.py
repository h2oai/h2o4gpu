from pogs.libs.cpu import pogsCPU
from pogs.solvers.base import BaseSolver


if not pogsCPU:
	print('\nWarning: Cannot create a POGS CPU Solver instance without linking Python module to a compiled POGS CPU libirary')
	print('> Setting pogs.SolverCPU=None')
	SolverCPU=None
else:
	class SolverCPU(object):
		def __init__(self, A, **kwargs):
			self.solver = BaseSolver(A,pogsCPU)
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