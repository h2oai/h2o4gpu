from lasso import Lasso
from lasso_path import LassoPath
from logistic import Logistic
from lp_eq import LpEq
from lp_ineq import LpIneq
from nonneg_l2 import NonNegL2
from svm import Svm 
import sys

def run_all(gpu=False, double_precision=False):

  print("\nLasso.")
  print "Solve time:\t{:.2e} seconds\n".format(Lasso(200,2000, gpu=gpu, double_precision=double_precision))

  print("\nLasso Path.")
  print "Solve time:\t{:.2e} seconds\n".format(LassoPath(200,1000, gpu=gpu, double_precision=double_precision))

  print("\nLogistic Regression.")
  print "Solve time:\t{:.2e} seconds\n".format(Logistic(1000,100, gpu=gpu, double_precision=double_precision))

  print("\nLinear Program in Equality Form.")
  print "Solve time:\t{:.2e} seconds\n".format(LpEq(1000,200, gpu=gpu, double_precision=double_precision))

  print("\nLinear Program in Inequality Form.")
  print "Solve time:\t{:.2e} seconds\n".format(LpIneq(1000,200, gpu=gpu, double_precision=double_precision))

  print("\nNon-Negative Least Squares.")
  print "Solve time:\t{:.2e} seconds\n".format(NonNegL2(1000,200, gpu=gpu, double_precision=double_precision))

  print("\nSupport Vector Machine.")
  print "Solve time:\t{:.2e} seconds\n".format(Svm(1000,200, gpu=gpu, double_precision=double_precision))


if __name__ == "__main__":
  gpu=False
  double=False
  if 'gpu' in sys.argv:
    gpu=True
  if 'double' in sys.argv:
    double=True
  run_all(gpu=gpu,double_precision=double)