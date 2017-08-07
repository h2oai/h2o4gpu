from ctypes import *
import numpy as np
import sys
from h2ogpuml.types import cptr
from h2ogpuml.libs.elastic_net_cpu import h2ogpumlGLMCPU
from h2ogpuml.libs.elastic_net_gpu import h2ogpumlGLMGPU
from h2ogpuml.solvers.utils import devicecount

"""
H2O GLM Solver

:param shared_a 
:param n_threads
:param n_gpus
:param ord
:param intercept
:param standardize
:param lambda_min_ratio
:param n_lambdas
:param n_folds
:param n_alphas
:param stop_early
:param stop_early_error_fraction
:param max_interations
:param verbose
:param family
"""

class GLM(object):
    # TODO shared_a and standardize do not work currently. Always need to set to 0.
    def __init__(self, shared_a=0, n_threads=None, n_gpus=-1, ord='r', intercept=1, standardize=0, lambda_min_ratio=1E-7,
                 n_lambdas=100, n_folds=1, n_alphas=1, stop_early=1, stop_early_error_fraction=1.0, max_iterations=5000, verbose=0, family = "elasticnet"):

        #TODO Add type checking

        n_gpus, device_count = devicecount(n_gpus)

        if n_threads == None:
            # not required number of threads, but normal.  Bit more optimal to use 2 threads for CPU, but 1 thread per GPU is optimal.
            n_threads = 1 if (n_gpus == 0) else n_gpus

        if not h2ogpumlGLMGPU:
            print(
                '\nWarning: Cannot create a H2OGPUML Elastic Net GPU Solver instance without linking Python module to a compiled H2OGPUML GPU library')
            print('> Use CPU or add CUDA libraries to $PATH and re-run setup.py\n\n')

        if not h2ogpumlGLMCPU:
            print(
                '\nWarning: Cannot create a H2OGPUML Elastic Net CPU Solver instance without linking Python module to a compiled H2OGPUML CPU library')
            print('> Use GPU or re-run setup.py\n\n')

        if ((n_gpus == 0) or (h2ogpumlGLMGPU is None) or (device_count == 0)):
            print("\nUsing CPU GLM solver %d %d\n" % (n_gpus, device_count))
            self.solver = _GLMBaseSolver(h2ogpumlGLMCPU, shared_a, n_threads, n_gpus, ord, intercept, standardize,
                                        lambda_min_ratio, n_lambdas, n_folds, n_alphas, stop_early, stop_early_error_fraction, max_iterations, verbose, family)
        else:
            if ((n_gpus > 0) or (h2ogpumlGLMGPU is None) or (device_count == 0)):
                print("\nUsing GPU GLM solver with %d GPUs\n" % n_gpus)
                self.solver = _GLMBaseSolver(h2ogpumlGLMGPU, shared_a, n_threads, n_gpus, ord, intercept, standardize,
                                            lambda_min_ratio, n_lambdas, n_folds, n_alphas, stop_early, stop_early_error_fraction, max_iterations, verbose, family)

        assert self.solver != None, "Couldn't instantiate GLM Solver"


    def upload_data(self, *args):
        return self.solver.upload_data(*args)

    def fit_ptr(self, source_dev, m_train, n, m_valid, precision, a, b, c, d, e, *args):
        return self.solver.fit_ptr(source_dev, m_train, n, m_valid, precision, a, b, c, d, e, *args)

    def fit(self, train_x, train_y, *args):
        return self.solver.fit(train_x, train_y, *args)

    def predict(self, valid_x, *args):
        return self.solver.predict(valid_x, *args)

    def predict_ptr(self, valid_xptr, *args):
        return self.solver.predict_ptr(valid_xptr, *args)

    def fit_predict(self, train_x, train_y, *args):
        return self.solver.fit_predict(train_x, train_y, *args)

    def fit_predict_ptr(self, source_dev, m_train, n, m_valid, precision, a, b, c, d, e, *args):
        return self.solver.fit_predict_ptr(source_dev, m_train, n, m_valid, precision, a, b, c, d, e, *args)

    #Define all properties of GLM class
    @property
    def get_tols(self):
        return self.solver.get_tols()

    @property
    def get_error(self):
        return self.solver.get_error()

    @property
    def get_lambdas(self):
        return self.solver.get_lambdas()

    @property
    def get_alphas(self):
        return self.solver.get_alphas()

    @property
    def free_data(self):
        return self.solver.free_data()

    @property
    def free_sols(self):
        return self.solver.free_sols()

    @property
    def free_preds(self):
        return self.solver.free_preds()

    @property
    def finish(self):
        return self.solver.finish()


class _GLMBaseSolver(object):
    class info:
        pass

    class solution:
        pass

    def __init__(self, lib, shared_a, n_threads, n_gpus, ordin, intercept, standardize, lambda_min_ratio, n_lambdas,
                 n_folds, n_alphas, stop_early, stop_early_error_fraction, max_iterations, verbose, family):
        assert lib and (lib == h2ogpumlGLMCPU or lib == h2ogpumlGLMGPU)
        self.lib = lib

        self.n = 0
        self.m_train = 0
        self.m_valid = 0

        self.n_gpus = n_gpus
        self.source_dev = 0  # assume Dev=0 is source of data for upload_data
        self.source_me = 0  # assume thread=0 is source of data for upload_data
        self.shared_a = shared_a
        self.n_threads = n_threads
        self.ord = ord(ordin)
        self.intercept = intercept
        self.standardize = standardize
        self.lambda_min_ratio = lambda_min_ratio
        self.n_lambdas = n_lambdas
        self.n_folds = n_folds
        self.n_alphas = n_alphas
        self.uploaded_data = 0
        self.did_fit_ptr = 0
        self.did_predict = 0
        self.stop_early=stop_early
        self.stop_early_error_fraction=stop_early_error_fraction
        self.max_iterations=max_iterations
        self.verbose=verbose
        self.family = ord(family.split()[0][0])

    def upload_data(self, source_dev, train_x, train_y, valid_x=None, valid_y=None, weight=None):
        if self.uploaded_data == 1:
            self.free_data()
        self.uploaded_data = 1
        #
        #################
        if train_x is not None:
            try:
                if (train_x.dtype == np.float64):
                    if self.verbose > 0:
                        print("Detected np.float64 train_x")
                    sys.stdout.flush()
                    self.double_precision1 = 1
                if (train_x.dtype == np.float32):
                    if self.verbose > 0:
                        print("Detected np.float32 train_x")
                    sys.stdout.flush()
                    self.double_precision1 = 0
            except:
                self.double_precision1 = -1
            try:
                if train_x.value is not None:
                    m_train = train_x.shape[0]
                    n1 = train_x.shape[1]
                else:
                    m_train = 0
                    n1 = -1
            except:
                m_train = train_x.shape[0]
                n1 = train_x.shape[1]
        else:
            m_train = 0
            n1 = -1
        self.m_train = m_train
        ################
        if valid_x is not None:
            try:
                if (valid_x.dtype == np.float64):
                    self.double_precision2 = 1
                if (valid_x.dtype == np.float32):
                    self.double_precision2 = 0
            except:
                self.double_precision2 = -1
            #
            try:
                if valid_x.value is not None:
                    m_valid = valid_x.shape[0]
                    n2 = valid_x.shape[1]
                else:
                    m_valid = 0
                    n2 = -1
            except:
                m_valid = valid_x.shape[0]
                n2 = valid_x.shape[1]
        else:
            m_valid = 0
            n2 = -1
            self.double_precision2 = -1
        self.m_valid = m_valid
        ################
        if train_y is not None:
            try:
                if (train_y.dtype == np.float64):
                    self.double_precision3 = 1
                if (train_y.dtype == np.float32):
                    self.double_precision3 = 0
            except:
                self.double_precision3 = -1
            #
            try:
                if train_y.value is not None:
                    m_train2 = train_y.shape[0]
                else:
                    m_train2 = 0
            except:
                m_train2 = train_y.shape[0]
        else:
            m_train2 = 0
            self.double_precision3 = -1
        ################
        if valid_y is not None:
            try:
                if (valid_y.dtype == np.float64):
                    self.double_precision4 = 1
                if (valid_y.dtype == np.float32):
                    self.double_precision4 = 0
            except:
                self.double_precision4 = -1
            #
            try:
                if valid_y.value is not None:
                    m_valid2 = valid_y.shape[0]
                else:
                    m_valid2 = 0
            except:
                m_valid2 = valid_y.shape[0]
        else:
            m_valid2 = 0
            self.double_precision4 = -1
        ################
        if weight is not None:
            try:
                if (weight.dtype == np.float64):
                    self.double_precision5 = 1
                if (weight.dtype == np.float32):
                    self.double_precision5 = 0
            except:
                self.double_precision5 = -1
            #
            try:
                if weight.value is not None:
                    m_train3 = weight.shape[0]
                else:
                    m_train3 = 0
            except:
                m_train3 = weight.shape[0]
        else:
            m_train3 = 0
            self.double_precision5 = -1
        ###############
        if self.double_precision1 >= 0 and self.double_precision2 >= 0:
            if (self.double_precision1 != self.double_precision2):
                print("train_x and valid_x must be same precision")
                exit(0)
            else:
                self.double_precision = self.double_precision1  # either one
        elif self.double_precision1 >= 0:
            self.double_precision = self.double_precision1
        elif self.double_precision2 >= 0:
            self.double_precision = self.double_precision2
        ###############
        if self.double_precision1 >= 0 and self.double_precision3 >= 0:
            if (self.double_precision1 != self.double_precision3):
                print("train_x and train_y must be same precision")
                exit(0)
        ###############
        if self.double_precision2 >= 0 and self.double_precision4 >= 0:
            if (self.double_precision2 != self.double_precision4):
                print("valid_x and valid_y must be same precision")
                exit(0)
        ###############
        if self.double_precision3 >= 0 and self.double_precision5 >= 0:
            if (self.double_precision3 != self.double_precision5):
                print("train_y and weight must be same precision")
                exit(0)
        ###############
        if n1 >= 0 and n2 >= 0:
            if (n1 != n2):
                print("train_x and valid_x must have same number of columns")
                exit(0)
            else:
                n = n1  # either one
        elif n1 >= 0:
            n = n1
        elif n2 >= 0:
            n = n2
        self.n = n
        ################
        a = c_void_p(0)
        b = c_void_p(0)
        c = c_void_p(0)
        d = c_void_p(0)
        e = c_void_p(0)
        if (self.double_precision == 1):
            null_ptr = POINTER(c_double)()
            #
            if train_x is not None:
                try:
                    if train_x.value is not None:
                        A = cptr(train_x, dtype=c_double)
                    else:
                        A = null_ptr
                except:
                    A = cptr(train_x, dtype=c_double)
            else:
                A = null_ptr
            if train_y is not None:
                try:
                    if train_y.value is not None:
                        B = cptr(train_y, dtype=c_double)
                    else:
                        B = null_ptr
                except:
                    B = cptr(train_y, dtype=c_double)
            else:
                B = null_ptr
            if valid_x is not None:
                try:
                    if valid_x.value is not None:
                        C = cptr(valid_x, dtype=c_double)
                    else:
                        C = null_ptr
                except:
                    C = cptr(valid_x, dtype=c_double)
            else:
                C = null_ptr
            if valid_y is not None:
                try:
                    if valid_y.value is not None:
                        D = cptr(valid_y, dtype=c_double)
                    else:
                        D = null_ptr
                except:
                    D = cptr(valid_y, dtype=c_double)
            else:
                D = null_ptr
            if weight is not None:
                try:
                    if weight.value is not None:
                        E = cptr(weight, dtype=c_double)
                    else:
                        E = null_ptr
                except:
                    E = cptr(weight, dtype=c_double)
            else:
                E = null_ptr
            status = self.lib.make_ptr_double(c_int(self.shared_a), c_int(self.source_me), c_int(source_dev),
                                              c_size_t(m_train), c_size_t(n), c_size_t(m_valid), c_int(self.ord),
                                              A, B, C, D, E, pointer(a), pointer(b), pointer(c), pointer(d), pointer(e))
        elif (self.double_precision == 0):
            if self.verbose>0:
                print("Detected np.float32")
                sys.stdout.flush()
            self.double_precision = 0
            null_ptr = POINTER(c_float)()
            #
            if train_x is not None:
                try:
                    if train_x.value is not None:
                        A = cptr(train_x, dtype=c_float)
                    else:
                        A = null_ptr
                except:
                    A = cptr(train_x, dtype=c_float)
            else:
                A = null_ptr
            if train_y is not None:
                try:
                    if train_y.value is not None:
                        B = cptr(train_y, dtype=c_float)
                    else:
                        B = null_ptr
                except:
                    B = cptr(train_y, dtype=c_float)
            else:
                B = null_ptr
            if valid_x is not None:
                try:
                    if valid_x.value is not None:
                        C = cptr(valid_x, dtype=c_float)
                    else:
                        C = null_ptr
                except:
                    C = cptr(valid_x, dtype=c_float)
            else:
                C = null_ptr
            if valid_y is not None:
                try:
                    if valid_y.value is not None:
                        D = cptr(valid_y, dtype=c_float)
                    else:
                        D = null_ptr
                except:
                    D = cptr(valid_y, dtype=c_float)
            else:
                D = null_ptr
            if weight is not None:
                try:
                    if weight.value is not None:
                        E = cptr(weight, dtype=c_float)
                    else:
                        E = null_ptr
                except:
                    E = cptr(weight, dtype=c_float)
            else:
                E = null_ptr
            status = self.lib.make_ptr_float(c_int(self.shared_a), c_int(self.source_me), c_int(source_dev),
                                             c_size_t(m_train), c_size_t(n), c_size_t(m_valid), c_int(self.ord),
                                             A, B, C, D, E, pointer(a), pointer(b), pointer(c), pointer(d), pointer(e))
        else:
            print("Unknown numpy type detected")
            print(train_x.dtype)
            sys.stdout.flush()
            return a, b, c, d, e

        assert status == 0, "Failure uploading the data"
        # print("a=",hex(a.value))
        # print("b=",hex(b.value))
        # print("c=",hex(c.value))
        # print("d=",hex(d.value))
        # print("e=",hex(e.value))
        self.solution.double_precision = self.double_precision
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        return a, b, c, d, e

    # source_dev here because generally want to take in any pointer, not just from our test code
    def fit_ptr(self, source_dev, m_train, n, m_valid, precision, a, b, c, d, e, give_full_path=0, do_predict=0, free_input_data=0, stop_early=None, stop_early_error_fraction=None, max_iterations=None, verbose=None):
        # store some things for later call to predict_ptr()
        self.source_dev = source_dev
        self.m_train = m_train
        self.n = n
        self.m_valid = m_valid
        self.precision = precision
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.give_full_path = give_full_path

        if stop_early is None:
            stop_early=self.stop_early
        if stop_early_error_fraction is None:
            stop_early_error_fraction=self.stop_early_error_fraction
        if max_iterations is None:
            max_iterations = self.max_iterations
        if verbose is None:
            verbose = self.verbose



        #print("a"); print(a)
        #print("b"); print(b)
        #print("c"); print(c)
        #print("d"); print(d)
        #print("e"); print(e)
        #sys.stdout.flush()


        ############
        if do_predict == 0 and self.did_fit_ptr == 1:
            self.free_sols()
        else:
            # otherwise don't clear solution, just use it
            pass
        ################
        self.did_fit_ptr = 1
        ###############
        # not calling with self.source_dev because want option to never use default but instead input pointers from foreign code's pointers
        if hasattr(self, 'double_precision'):
            which_precision = self.double_precision
        else:
            which_precision = precision
            self.double_precision = precision
        ##############
        if do_predict == 0:
            # initialize if doing fit
            x_vs_alpha_lambda = c_void_p(0)
            x_vs_alpha = c_void_p(0)
            valid_pred_vs_alpha_lambda = c_void_p(0)
            valid_pred_vs_alpha = c_void_p(0)
            count_full = c_size_t(0)
            count_short = c_size_t(0)
            count_more = c_size_t(0)
        else:
            # restore if predict
            x_vs_alpha_lambda = self.x_vs_alpha_lambda
            x_vs_alpha = self.x_vs_alpha
            valid_pred_vs_alpha_lambda = self.valid_pred_vs_alpha_lambda
            valid_pred_vs_alpha = self.valid_pred_vs_alpha
            count_full = self.count_full
            count_short = self.count_short
            count_more = self.count_more
        ################
        #
        c_size_t_p = POINTER(c_size_t)
        if (which_precision == 1):
            self.mydtype = np.double
            self.myctype = c_double
            if verbose>0:
                print("double precision fit")
                sys.stdout.flush()
            self.lib.elastic_net_ptr_double(
                c_int(self.family),
                c_int(do_predict),
                c_int(source_dev), c_int(1), c_int(self.shared_a), c_int(self.n_threads), c_int(self.n_gpus),
                c_int(self.ord),
                c_size_t(m_train), c_size_t(n), c_size_t(m_valid), c_int(self.intercept), c_int(self.standardize),
                c_double(self.lambda_min_ratio), c_int(self.n_lambdas), c_int(self.n_folds), c_int(self.n_alphas),
                c_int(stop_early), c_double(stop_early_error_fraction), c_int(max_iterations), c_int(verbose),
                a, b, c, d, e
                , give_full_path
                , pointer(x_vs_alpha_lambda), pointer(x_vs_alpha)
                , pointer(valid_pred_vs_alpha_lambda), pointer(valid_pred_vs_alpha)
                , cast(addressof(count_full), c_size_t_p), cast(addressof(count_short), c_size_t_p),
                cast(addressof(count_more), c_size_t_p)
            )
        else:
            self.mydtype = np.float
            self.myctype = c_float
            if verbose>0:
                print("single precision fit")
                sys.stdout.flush()
            self.lib.elastic_net_ptr_float(
                c_int(self.family),
                c_int(do_predict),
                c_int(source_dev), c_int(1), c_int(self.shared_a), c_int(self.n_threads), c_int(self.n_gpus),
                c_int(self.ord),
                c_size_t(m_train), c_size_t(n), c_size_t(m_valid), c_int(self.intercept), c_int(self.standardize),
                c_double(self.lambda_min_ratio), c_int(self.n_lambdas), c_int(self.n_folds), c_int(self.n_alphas),
                c_int(stop_early), c_double(stop_early_error_fraction), c_int(max_iterations), c_int(verbose),
                a, b, c, d, e
                , give_full_path
                , pointer(x_vs_alpha_lambda), pointer(x_vs_alpha)
                , pointer(valid_pred_vs_alpha_lambda), pointer(valid_pred_vs_alpha)
                , cast(addressof(count_full), c_size_t_p), cast(addressof(count_short), c_size_t_p),
                cast(addressof(count_more), c_size_t_p)
            )
        #
        # if should or user wanted to save or free data, do that now that we are done using a,b,c,d,e
        # This means have to upload_data() again before fit_ptr or predict_ptr or only call fit and predict
        if free_input_data==1:
            self.free_data()
        #####################################
        # PROCESS OUTPUT
        # save pointers
        self.x_vs_alpha_lambda = x_vs_alpha_lambda
        self.x_vs_alpha = x_vs_alpha
        self.valid_pred_vs_alpha_lambda = valid_pred_vs_alpha_lambda
        self.valid_pred_vs_alpha = valid_pred_vs_alpha
        self.count_full = count_full
        self.count_short = count_short
        self.count_more = count_more
        #
        count_full_value = count_full.value
        count_short_value = count_short.value
        count_more_value = count_more.value
        # print("counts=%d %d %d" % (count_full_value,count_short_value,count_more_value))
        ######
        if give_full_path == 1:
            num_all = int(count_full_value / (self.n_alphas * self.n_lambdas))
        else:
            num_all = int(count_short_value / (self.n_alphas))
        #
        NUMALLOTHER = num_all - n
        NUMERROR = 3  # should be consistent with src/common/elastic_net_ptr.cpp
        NUMOTHER = NUMALLOTHER - NUMERROR
        if NUMOTHER != 3:
            print("NUMOTHER=%d but expected 3" % (NUMOTHER))
            print("count_full_value=%d count_short_value=%d count_more_value=%d num_all=%d NUMALLOTHER=%d" % (
            int(count_full_value), int(count_short_value), int(count_more_value), int(num_all), int(NUMALLOTHER)))
            sys.stdout.flush()
            exit(0)
        #
        if give_full_path == 1 and do_predict == 0:
                # x_vs_alpha_lambda contains solution (and other data) for all lambda and alpha
                self.x_vs_alpha_lambdanew = np.fromiter(cast(x_vs_alpha_lambda, POINTER(self.myctype)), dtype=self.mydtype,
                                                     count=count_full_value)
                self.x_vs_alpha_lambdanew = np.reshape(self.x_vs_alpha_lambdanew, (self.n_lambdas, self.n_alphas, num_all))
                self.x_vs_alpha_lambdapure = self.x_vs_alpha_lambdanew[:, :, 0:n]
                self.error_vs_alpha_lambda = self.x_vs_alpha_lambdanew[:, :, n:n + NUMERROR]
                self.lambdas = self.x_vs_alpha_lambdanew[:, :, n + NUMERROR:n + NUMERROR + 1]
                self.alphas = self.x_vs_alpha_lambdanew[:, :, n + NUMERROR + 1:n + NUMERROR + 2]
                self.tols = self.x_vs_alpha_lambdanew[:, :, n + NUMERROR + 2:n + NUMERROR + 3]
                #
                self.solution.x_vs_alpha_lambdapure = self.x_vs_alpha_lambdapure
                self.info.error_vs_alpha_lambda = self.error_vs_alpha_lambda
                self.info.lambdas = self.lambdas
                self.info.alphas = self.alphas
                self.info.tols = self.tols
            #
        if give_full_path==1 and do_predict==1:
            thecount = int(count_full_value / (n + NUMALLOTHER) * m_valid)
            self.valid_pred_vs_alpha_lambdanew = np.fromiter(cast(valid_pred_vs_alpha_lambda, POINTER(self.myctype)),
                                                          dtype=self.mydtype, count=thecount)
            self.valid_pred_vs_alpha_lambdanew = np.reshape(self.valid_pred_vs_alpha_lambdanew,
                                                         (self.n_lambdas, self.n_alphas, m_valid))
            self.valid_pred_vs_alpha_lambdapure = self.valid_pred_vs_alpha_lambdanew[:, :, 0:m_valid]
            #
        if do_predict == 0: # give_full_path==0 or 1
            # x_vs_alpha contains only best of all lambda for each alpha
            self.x_vs_alphanew = np.fromiter(cast(x_vs_alpha, POINTER(self.myctype)), dtype=self.mydtype,
                                           count=count_short_value)
            self.x_vs_alphanew = np.reshape(self.x_vs_alphanew, (self.n_alphas, num_all))
            self.x_vs_alphapure = self.x_vs_alphanew[:, 0:n]
            self.error_vs_alpha = self.x_vs_alphanew[:, n:n + NUMERROR]
            self.lambdas2 = self.x_vs_alphanew[:, n + NUMERROR:n + NUMERROR + 1]
            self.alphas2 = self.x_vs_alphanew[:, n + NUMERROR + 1:n + NUMERROR + 2]
            self.tols2 = self.x_vs_alphanew[:, n + NUMERROR + 2:n + NUMERROR + 3]
            #
            self.solution.x_vs_alphapure = self.x_vs_alphapure
            self.info.error_vs_alpha = self.error_vs_alpha
            self.info.lambdas2 = self.lambdas2
            self.info.alphas2 = self.alphas2
            self.info.tols2 = self.tols2
        #
        if give_full_path==0 and do_predict == 1: # preds exclusively operate for x_vs_alpha or x_vs_alpha_lambda
            thecount = int(count_short_value / (n + NUMALLOTHER) * m_valid)
            if verbose>0:
                print("thecount=%d count_full_value=%d count_short_value=%d n=%d NUMALLOTHER=%d m_valid=%d" % (
                    thecount, count_full_value, count_short_value, n, NUMALLOTHER, m_valid))
                sys.stdout.flush()
            self.valid_pred_vs_alphanew = np.fromiter(cast(valid_pred_vs_alpha, POINTER(self.myctype)), dtype=self.mydtype,
                                                    count=thecount)
            self.valid_pred_vs_alphanew = np.reshape(self.valid_pred_vs_alphanew, (self.n_alphas, m_valid))
            self.valid_pred_vs_alphapure = self.valid_pred_vs_alphanew[:, 0:m_valid]
        #
        #######################
        # return numpy objects
        if do_predict == 0:
            self.did_predict = 0
            if give_full_path == 1:
                return (self.x_vs_alpha_lambdapure, self.x_vs_alphapure)
            else:
                return (None, self.x_vs_alphapure)
        else:
            self.did_predict = 1
            if give_full_path == 1:
                return self.valid_pred_vs_alpha_lambdapure
            else:
                return self.valid_pred_vs_alphapure

    def fit(self, train_x, train_y, valid_x=None, valid_y=None, weight=None, give_full_path=0, do_predict=0, free_input_data=1, stop_early=None, stop_early_error_fraction=None, max_iterations=None, verbose=None):
        #
        self.give_full_path = give_full_path
        ################
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.weight = weight
        #
        if stop_early is None:
            stop_early=self.stop_early
        if stop_early_error_fraction is None:
            stop_early_error_fraction=self.stop_early_error_fraction
        if max_iterations is None:
            max_iterations = self.max_iterations
        if verbose is None:
            verbose = self.verbose
        ##############
        if train_x is not None:
            try:
                if train_x.value is not None:
                    # get shapes
                    shape_x = np.shape(train_x)
                    m_train = shape_x[0]
                    n1 = shape_x[1]
                else:
                    if verbose > 0:
                        print("no train_x")
                    n1 = -1
            except:
                # get shapes
                shape_x = np.shape(train_x)
                m_train = shape_x[0]
                n1 = shape_x[1]
        else:
            if verbose>0:
                print("no train_x")
            m_train = 0
            n1 = -1
        #############
        if train_y is not None:
            try:
                if train_y.value is not None:
                    # get shapes
                    if verbose > 0:
                        print("Doing fit")
                    shape_y = np.shape(train_y)
                    m_y = shape_y[0]
                    if (m_train != m_y):
                        print("training X and Y must have same number of rows, but m_train=%d m_y=%d\n" % (m_train, m_y))
                else:
                    m_y = -1
            except:
                # get shapes
                if verbose > 0:
                    print("Doing fit")
                shape_y = np.shape(train_y)
                m_y = shape_y[0]
                if (m_train != m_y):
                    print("training X and Y must have same number of rows, but m_train=%d m_y=%d\n" % (m_train, m_y))
        else:
            if verbose>0:
                print("Doing predict")
            m_y = -1
        ###############
        if valid_x is not None:
            try:
                if valid_x.value is not None:
                    shapevalid_x = np.shape(valid_x)
                    m_valid = shapevalid_x[0]
                    n2 = shapevalid_x[1]
                else:
                    if verbose>0:
                        print("no valid_x")
                    m_valid = 0
                    n2 = -1
            except:
                shapevalid_x = np.shape(valid_x)
                m_valid = shapevalid_x[0]
                n2 = shapevalid_x[1]
        else:
            if verbose>0:
                print("no valid_x")
            m_valid = 0
            n2 = -1
        if verbose > 0:
            print("m_valid=%d" % (m_valid))
        sys.stdout.flush()
        ###############
        if valid_y is not None:
            try:
                if valid_y.value is not None:
                    shapevalid_y = np.shape(valid_y)
                    m_valid_y = shapevalid_y[0]
                else:
                    if verbose > 0:
                        print("no valid_y")
                    m_valid_y = -1
            except:
                shapevalid_y = np.shape(valid_y)
                m_valid_y = shapevalid_y[0]
        else:
            if verbose>0:
                print("no valid_y")
            m_valid_y = -1
        ################
        # check do_predict input
        if do_predict == 0:
            if verbose>0:
                if n1 >= 0 and m_y >= 0:
                    print("Correct train inputs")
                else:
                    print("Incorrect train inputs")
                    exit(0)
        if do_predict == 1:
            if (n1 == -1 and n2 >= 0 and m_valid_y == -1 and m_y == -1) or (n1 == -1 and n2 >= 0 and m_y == -1):
                if verbose > 0:
                    print("Correct prediction inputs")
            else:
                print("Incorrect prediction inputs")
                exit(0)
        #################
        if do_predict == 0:
            if (n1 >= 0 and n2 >= 0 and n1 != n2):
                print("train_x and valid_x must have same number of columns, but n=%d n2=%d\n" % (n1, n2))
                exit(0)
            else:
                n = n1  # either
        else:
            n = n2  # pick valid_x
        ##################
        if do_predict == 0:
            if (m_valid >= 0 and m_valid_y >= 0 and m_valid != m_valid_y):
                print("valid_x and valid_y must have same number of rows, but m_valid=%d m_valid_y=%d\n" % (m_valid, m_valid_y))
                exit(0)
        else:
            # otherwise m_valid is used, and m_valid_y can be there or not (sets whether do error or not)
            pass
        #################
        if do_predict == 0:
            if ((m_valid==0 or m_valid==-1) and n2>0) or (m_valid>0 and (n2==0 or n2==-1)):
            #if ((valid_x is not None and valid_y == None) or (valid_x == None and valid_y is not None)):
                print(
                    "Must input both valid_x and valid_y or neither.")  # TODO FIXME: Don't need valid_y if just want preds and no error, but don't return error in fit, so leave for now
                exit(0)
                #
        ##############
        source_dev = 0  # assume GPU=0 is fine as source
        a, b, c, d, e = self.upload_data(source_dev, train_x, train_y, valid_x, valid_y, weight)
        precision = 0  # won't be used
        self.fit_ptr(source_dev, m_train, n, m_valid, precision, a, b, c, d, e, give_full_path, do_predict=do_predict, free_input_data=free_input_data, stop_early=stop_early, stop_early_error_fraction=stop_early_error_fraction, max_iterations=max_iterations, verbose=verbose)
        if do_predict == 0:
            if give_full_path == 1:
                return (self.x_vs_alpha_lambdapure, self.x_vs_alphapure)
            else:
                return (None, self.x_vs_alphapure)
        else:
            if give_full_path == 1:
                return (self.valid_pred_vs_alpha_lambdapure, self.valid_pred_vs_alphapure)
            else:
                return (None, self.valid_pred_vs_alphapure)

    def get_error(self):
        if self.give_full_path==1:
            return (self.error_vs_alpha_lambda, self.error_vs_alpha)
        else:
            return (None, self.error_vs_alpha)

    def get_lambdas(self):
        if self.give_full_path==1:
            return (self.lambdas, self.lambdas2)
        else:
            return (None, self.lambdas2)

    def get_alphas(self):
        if self.give_full_path==1:
            return (self.alphas, self.alphas2)
        else:
            return (None, self.alphas2)

    def get_tols(self):
        if self.give_full_path==1:
            return (self.tols, self.tols2)
        else:
            return (None, self.tols2)

    def predict(self, valid_x, valid_y=None, testweight=None, give_full_path=0, free_input_data=1):
        # if pass None train_x and train_y, then do predict using valid_x and weight (if given)
        # unlike upload_data and fit_ptr (and so fit) don't free-up predictions since for single model might request multiple predictions.  User has to call finish themselves to cleanup.
        do_predict = 1
        if give_full_path==1:
            self.prediction_full = self.fit(None, None, valid_x, valid_y, testweight, give_full_path, do_predict, free_input_data)
        else:
            self.prediction_full = None
        self.prediction = self.fit(None, None, valid_x, valid_y, testweight, 0, do_predict, free_input_data)
        return (self.prediction_full, self.prediction)  # something like valid_y

    def predict_ptr(self, valid_xptr, valid_yptr=None, give_full_path=0, free_input_data=0):
        do_predict = 1
        #print("%d %d %d %d %d" % (self.source_dev, self.m_train, self.n, self.m_valid, self.precision)) ; sys.stdout.flush()
        self.prediction = self.fit_ptr(self.source_dev, self.m_train, self.n, self.m_valid, self.precision, self.a, self.b,
                                      valid_xptr, valid_yptr, self.e, 0, do_predict, free_input_data)
        if give_full_path==1: # then need to run twice
            self.prediction_full = self.fit_ptr(self.source_dev, self.m_train, self.n, self.m_valid, self.precision, self.a, self.b, valid_xptr, valid_yptr, self.e, give_full_path, do_predict, free_input_data)
        else:
            self.prediction_full = None
        return (self.prediction_full, self.prediction)  # something like valid_y

    def fit_predict(self, train_x, train_y, valid_x=None, valid_y=None, weight=None, give_full_path=0, free_input_data=1, stop_early=None, stop_early_error_fraction=None, max_iterations=None, verbose=None):
        if stop_early is None:
            stop_early=self.stop_early
        if stop_early_error_fraction is None:
            stop_early_error_fraction=self.stop_early_error_fraction
        if max_iterations is None:
            max_iterations = self.max_iterations
        if verbose is None:
            verbose = self.verbose
        do_predict = 0  # only fit at first
        self.fit(train_x, train_y, valid_x, valid_y, weight, give_full_path, do_predict, free_input_data=0, stop_early=stop_early, stop_early_error_fraction=stop_early_error_fraction, max_iterations=max_iterations, verbose=verbose)
        if valid_x == None:
            if give_full_path==1:
                self.prediction_full = self.predict(train_x, train_y, testweight=weight, give_full_path=give_full_path, free_input_data=free_input_data)
            else:
                self.prediction_full = None
            self.prediction = self.predict(train_x, train_y, testweight=weight, give_full_path=0,
                                       free_input_data=free_input_data)
        else:
            if give_full_path==1:
                self.prediction_full = self.predict(valid_x, valid_y, testweight=weight, give_full_path=give_full_path, free_input_data=free_input_data)
            else:
                self.prediction_full = None
            self.prediction = self.predict(valid_x, valid_y, testweight=weight, give_full_path=0,
                                           free_input_data=free_input_data)
        return (self.prediction_full, self.prediction)

    def fit_predict_ptr(self, source_dev, m_train, n, m_valid, precision, a, b, c, d, e, give_full_path=0, free_input_data=0, stop_early=None, stop_early_error_fraction=None, max_iterations=None, verbose=None):
        do_predict = 0  # only fit at first
        if stop_early is None:
            stop_early=self.stop_early
        if stop_early_error_fraction is None:
            stop_early_error_fraction=self.stop_early_error_fraction
        if max_iterations is None:
            max_iterations = self.max_iterations
        if verbose is None:
            verbose = self.verbose
        self.fit_ptr(source_dev, m_train, n, m_valid, precision, a, b, c, d, e, give_full_path, do_predict, free_input_data=0, stop_early=stop_early, stop_early_error_fraction=stop_early_error_fraction, max_iterations=max_iterations, verbose=verbose)
        if c is None or c is c_void_p(0):
            self.prediction = self.predict_ptr(a, b, 0, free_input_data=free_input_data)
            if give_full_path==1:
                self.prediction_full = self.predict_ptr(a, b, give_full_path, free_input_data=free_input_data)
            else:
                self.prediction_full = None
        else:
            self.prediction = self.predict_ptr(c, d, 0, free_input_data=free_input_data)
            if give_full_path==1:
                self.prediction_full = self.predict_ptr(c, d, give_full_path, free_input_data=free_input_data)
            else:
                self.prediction_full = None
        return (self.prediction_full, self.prediction)

    def free_data(self):
        # NOTE: For now, these are automatically freed when done with fit -- ok, since not used again
        if self.uploaded_data == 1:
            self.uploaded_data = 0
            if self.double_precision == 1:
                self.lib.modelfree1_double(self.a)
                self.lib.modelfree1_double(self.b)
                self.lib.modelfree1_double(self.c)
                self.lib.modelfree1_double(self.d)
                self.lib.modelfree1_double(self.e)
            else:
                self.lib.modelfree1_float(self.a)
                self.lib.modelfree1_float(self.b)
                self.lib.modelfree1_float(self.c)
                self.lib.modelfree1_float(self.d)
                self.lib.modelfree1_float(self.e)

    def free_sols(self):
        if self.did_fit_ptr == 1:
            self.did_fit_ptr = 0
            if self.double_precision == 1:
                self.lib.modelfree2_double(self.x_vs_alpha_lambda)
                self.lib.modelfree2_double(self.x_vs_alpha)
            else:
                self.lib.modelfree2_float(self.x_vs_alpha_lambda)
                self.lib.modelfree2_float(self.x_vs_alpha)

    def free_preds(self):
        if self.did_predict == 1:
            self.did_predict = 0
            if self.double_precision == 1:
                self.lib.modelfree2_double(self.valid_pred_vs_alpha_lambda)
                self.lib.modelfree2_double(self.valid_pred_vs_alpha)
            else:
                self.lib.modelfree2_float(self.valid_pred_vs_alpha_lambda)
                self.lib.modelfree2_float(self.valid_pred_vs_alpha)

    def finish(self):
        self.free_data()
        self.free_sols()
        self.free_preds()
