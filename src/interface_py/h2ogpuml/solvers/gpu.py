from h2ogpuml.libs.gpu import h2ogpumlGPU
from h2ogpuml.solvers.base import BaseSolver

if not h2ogpumlGPU:
	print('\nWarning: Cannot create a H2OGPUML GPU Solver instance without linking Python module to a compiled H2OGPUML GPU library')
	print('> Setting h2ogpuml.SolverGPU=None')
	print('> Use h2ogpuml.SolverCPU(args...) or add CUDA libraries to $PATH and re-run setup.py\n\n')
	SolverGPU=None
else:
	class SolverGPU(object):
		def __init__(self, A, **kwargs):
			self.solver = BaseSolver(A,h2ogpumlGPU)
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
