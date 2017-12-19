# -*- encoding: utf-8 -*-
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
from ctypes import c_int, c_float, c_double, pointer
from numpy import ndarray
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from h2o4gpu.types import H2OConstants, cptr, make_settings, \
    make_solution, make_info, change_settings, change_solution, \
    Solution, FunctionVector
from h2o4gpu.libs.lib_pogs import CPUlib, GPUlib
from ..util.gpu import device_count

# TODO: catch Ctrl-C


class Pogs(object):
    """POGS solver"""

    def __init__(self, A, **kwargs):

        # Try to use all GPUs by default
        n_gpus = -1

        for key, value in kwargs.items():
            if key == "n_gpus":
                n_gpus = value
                # print("The value of {} is {}".format(key, value))
                # sys.stdout.flush()

        n_gpus, devices = device_count(n_gpus=n_gpus)

        gpu_lib = GPUlib().get()
        cpu_lib = CPUlib().get()

        if (n_gpus == 0) or (gpu_lib is None) or (devices == 0):
            print("\nUsing CPU GLM solver\n")
            self.solver = BaseSolver(A, cpu_lib)
        else:
            print("\nUsing GPU GLM solver with %d GPUs\n" % n_gpus)
            self.solver = BaseSolver(A, gpu_lib)

        assert self.solver is not None, "Couldn't instantiate Pogs Solver"

        self.info = self.solver.info
        self.solution = self.solver.pysolution

    def init(self, A, **kwargs):
        self.solver.init(A, **kwargs)

    def fit(self, f, g, **kwargs):
        self.solver.fit(f, g, **kwargs)

    def finish(self):
        self.solver.finish()

    def __delete__(self, instance):
        self.solver.finish()


class BaseSolver(object):
    """Solver class calling underlying POGS implementation"""

    def __init__(self, A, lib, **kwargs):
        try:
            self.dense = isinstance(A, ndarray) and len(A.shape) == 2
            self.CSC = isinstance(A, csc_matrix)
            self.CSR = isinstance(A, csr_matrix)

            assert self.dense or self.CSC or self.CSR
            assert A.dtype == c_float or A.dtype == c_double

            self.m = A.shape[0]
            self.n = A.shape[1]
            self.A = A
            self.lib = lib
            self.wDev = 0

            self.double_precision = A.dtype == c_double
            self.settings = make_settings(self.double_precision, **kwargs)
            self.pysolution = Solution(self.double_precision, self.m, self.n)
            self.solution = make_solution(self.pysolution)
            self.info = make_info(self.double_precision)
            self.order = H2OConstants.ROW_MAJ if (self.CSR or self.dense) \
                else H2OConstants.COL_MAJ

            if self.dense and not self.double_precision:
                self.work = self.lib.h2o4gpu_init_dense_single(
                    self.wDev, self.order, self.m, self.n, cptr(A, c_float))
            elif self.dense:
                self.work = self.lib.h2o4gpu_init_dense_double(
                    self.wDev, self.order, self.m, self.n, cptr(A, c_double))
            elif not self.double_precision:
                self.work = self.lib.h2o4gpu_init_sparse_single(
                    self.wDev, self.order, self.m, self.n, A.nnz,
                    cptr(A.data, c_float), cptr(A.indices, c_int),
                    cptr(A.indptr, c_int))
            else:
                self.work = self.lib.h2o4gpu_init_sparse_double(
                    self.wDev, self.order, self.m, self.n, A.nnz,
                    cptr(A.data, c_double), cptr(A.indices, c_int),
                    cptr(A.indptr, c_int))

        except AssertionError:
            print("Data must be a (m x n) numpy ndarray or scipy csc_matrix "
                  "containing float32 or float64 values")

    def init(self, A, lib, **kwargs):
        if not self.work:
            self.__init__(A, lib, **kwargs)
        else:
            print("H2O4GPU_work data structure already intialized,"
                  "cannot re-initialize without calling finish()")

    def fit(self, f, g, **kwargs):
        """ Fit """
        try:
            # assert f,g types
            assert isinstance(f, FunctionVector)
            assert isinstance(g, FunctionVector)

            # assert f,g lengths
            assert f.length() == self.m
            assert g.length() == self.n

            # pass previous rho through, if not first run (rho=0)
            if self.info.rho > 0:
                self.settings.rho = self.info.rho

            # apply user inputs
            change_settings(self.settings, **kwargs)
            change_solution(self.pysolution, **kwargs)

            if not self.work:
                print("No viable H2O4GPU_work pointer to call solve()."
                      "Call Solver.init( args... ) first")
                return
            elif not self.double_precision:
                self.lib.h2o4gpu_solve_single(self.work, pointer(self.settings),
                                              pointer(self.solution),
                                              pointer(self.info),
                                              cptr(f.a, c_float),
                                              cptr(f.b, c_float),
                                              cptr(f.c, c_float),
                                              cptr(f.d, c_float),
                                              cptr(f.e, c_float),
                                              cptr(f.h, c_int),
                                              cptr(g.a, c_float),
                                              cptr(g.b, c_float),
                                              cptr(g.c, c_float),
                                              cptr(g.d, c_float),
                                              cptr(g.e, c_float),
                                              cptr(g.h, c_int))
            else:
                self.lib.h2o4gpu_solve_double(self.work, pointer(self.settings),
                                              pointer(self.solution),
                                              pointer(self.info),
                                              cptr(f.a, c_double),
                                              cptr(f.b, c_double),
                                              cptr(f.c, c_double),
                                              cptr(f.d, c_double),
                                              cptr(f.e, c_double),
                                              cptr(f.h, c_int),
                                              cptr(g.a, c_double),
                                              cptr(g.b, c_double),
                                              cptr(g.c, c_double),
                                              cptr(g.d, c_double),
                                              cptr(g.e, c_double),
                                              cptr(g.h, c_int))

        except AssertionError:
            print("\nf and g must be objects of type FunctionVector with:")
            print(">length of f = m, # of rows in solver's data matrix A")
            print(">length of g = n, # of columns in solver's data matrix A")

    def finish(self):
        """Finish all work the solver was performing."""
        if not self.work:
            print("No viable H2O4GPU_work pointer to call finish()."
                  "Call Solver.init( args... ) first")
        elif not self.double_precision:
            self.lib.h2o4gpu_finish_single(self.work)
            self.work = None
        else:
            self.lib.h2o4gpu_finish_double(self.work)
            self.work = None
        print("shutting down... H2O4GPU_work freed in C++")

    def __delete__(self, instance):
        self.finish()
