from h2oaiglm.libs.cpu import h2oaiglmCPU
from h2oaiglm.solvers.base import BaseSolver


if not h2oaiglmCPU:
	print('\nWarning: Cannot create a H2OAIGLM CPU Solver instance without linking Python module to a compiled H2OAIGLM CPU library')
	print('> Setting h2oaiglm.SolverCPU=None')
	SolverCPU=None
else:
	class SolverCPU(object):
		def __init__(self, A, **kwargs):
			self.solver = BaseSolver(A,h2oaiglmCPU)
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
