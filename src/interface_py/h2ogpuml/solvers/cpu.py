from h2ogpuml.libs.cpu import h2ogpumlCPU
from h2ogpuml.solvers.base import BaseSolver


if not h2ogpumlCPU:
	print('\nWarning: Cannot create a H2OGPUML CPU Solver instance without linking Python module to a compiled H2OGPUML CPU library')
	print('> Setting h2ogpuml.SolverCPU=None')
	SolverCPU=None
else:
	class SolverCPU(object):
		def __init__(self, A, **kwargs):
			self.solver = BaseSolver(A,h2ogpumlCPU)
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
