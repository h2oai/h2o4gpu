__version__ = "0.0.1"
from pogs.types import FUNCTION, STATUS, FunctionVector
from pogs.solvers.cpu import SolverCPU
from pogs.solvers.gpu import SolverGPU
print("importing elastic_net_cpu")
from pogs.solvers.elastic_net_cpu import ElasticNetSolverCPU
print(ElasticNetSolverCPU!=None)
print("importing elastic_net_gpu")
from pogs.solvers.elastic_net_gpu import ElasticNetSolverGPU
print(ElasticNetSolverGPU!=None)
print("done")
