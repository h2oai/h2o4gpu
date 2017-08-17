from lasso import lasso
from lasso_path import lasso_path
from logistic import logistic
from lp_eq import lp_eq
from lp_ineq import lp_ineq
from nonneg_l2 import non_neg_l2
from svm import svm
import sys


def run_all(gpu=False, double_precision=False, m=1000, n=1000):
    print("\nLasso.")
    print("Solve time:\t{:.2e} seconds\n".format(lasso(m, n, gpu=gpu, double_precision=double_precision)))

    print("\nLasso Path.")
    print("Solve time:\t{:.2e} seconds\n".format(lasso_path(m, n, gpu=gpu, double_precision=double_precision)))

    print("\nLogistic Regression.")
    print("Solve time:\t{:.2e} seconds\n".format(logistic(m, n, gpu=gpu, double_precision=double_precision)))

    print("\nLinear Program in Equality Form.")
    print("Solve time:\t{:.2e} seconds\n".format(lp_eq(m, n, gpu=gpu, double_precision=double_precision)))

    print("\nLinear Program in Inequality Form.")
    print("Solve time:\t{:.2e} seconds\n".format(lp_ineq(m, n, gpu=gpu, double_precision=double_precision)))

    print("\nNon-Negative Least Squares.")
    print("Solve time:\t{:.2e} seconds\n".format(non_neg_l2(m, n, gpu=gpu, double_precision=double_precision)))

    print("\nSupport Vector Machine.")
    print("Solve time:\t{:.2e} seconds\n".format(svm(m, n, gpu=gpu, double_precision=double_precision)))


if __name__ == "__main__":
    gpu = True
    double = False
    m = 10000
    n = 10000
    if 'gpu' in sys.argv:
        gpu = True
    if 'double' in sys.argv:
        double = True
    run_all(gpu=gpu, double_precision=double, m=m, n=n)
