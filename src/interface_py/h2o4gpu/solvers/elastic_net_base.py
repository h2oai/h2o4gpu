#!/usr/bin/python
# -*- coding: utf-8 -*-
from ctypes import *
import numpy as np
import sys
from h2o4gpu.types import cptr
from h2o4gpu.libs.elastic_net_cpu import h2o4gpuGLMCPU
from h2o4gpu.libs.elastic_net_gpu import h2o4gpuGLMGPU
from h2o4gpu.solvers.utils import device_count, _get_data, _check_data_size, _convert_to_ptr, _check_equal
from h2o4gpu.util.typechecks import assert_is_type
import pandas as pd
from tabulate import tabulate

"""
H2O GLM Solver

:param int n_threads: Number of threads to use in the gpu. Default is None.
:param int n_gpus: Number of gpu's to use in GLM solver. Default is -1.
:param str order: Row major or Column major for C/C++ backend. Default is Row major ('r'). Must be "r" (Row major) or "c" (Column major).
:param bool fit_intercept: Include constant term in the model. Default is True.
:param int lambda_min_ratio: Minimum lambda used in lambda search. Default is 1e-7.
:param int n_lambdas: Number of lambdas to be used in a search. Default is 100.
:param int n_folds: Number of cross validation folds. Default is 1.
:param int n_alphas: Number of alphas to be used in a search. Default is 5.
:param float tol: tolerance.  Default is 1E-2.
:param bool lambda_stop_early: Stop early when there is no more relative improvement on train or validation. Default is True.
:param bool glm_stop_early: Stop early when there is no more relative improvement in the primary and dual residuals for ADMM.  Default is True
:param float glm_stop_early_error_fraction: Relative tolerance for metric-based stopping criterion (stop if relative improvement is not at least this much). Default is 1.0.
:param int max_iter: Maximum number of iterations. Default is 5000
:param int verbose: Print verbose information to the console if set to > 0. Default is 0.
:param str family: Use "logistic" for classification with logistic regression. Defaults to "elasticnet" for regression. Must be "logistic" or "elasticnet".
:param lambda_max: Maximum Lambda value to use.  Default is None, and then internally compute standard maximum
:param alpha_max: Maximum alpha.  Default is 1.0.
:param alpha_min: Minimum alpha.  Default is 0.0.
:param order: Order of data.  Default is None, and internally determined whether row 'r' or column 'c' major order.
"""


class GLM(object):
    class info:

        pass

    class solution:

        pass

    # TODO: add gpu_id like kmeans and ensure wraps around device_count
    def __init__(
            self,
            n_threads=None,
            n_gpus=-1,
            fit_intercept=True,
            lambda_min_ratio=1E-7,
            n_lambdas=100,
            n_folds=5,
            n_alphas=5,
            tol=1E-2,
            lambda_stop_early=True,
            glm_stop_early=True,
            glm_stop_early_error_fraction=1.0,
            max_iter=5000,
            verbose=0,
            family='elasticnet',
            give_full_path=0,
            lambda_max=None,
            alpha_max=1.0,
            alpha_min=0.0,
            order = None,
    ):

        # Type Checking

        assert_is_type(n_threads, int, None)
        assert_is_type(n_gpus, int)
        assert_is_type(fit_intercept, bool)
        assert_is_type(n_lambdas, int)
        assert_is_type(n_folds, int)
        assert_is_type(n_alphas, int)
        assert_is_type(tol, float)
        assert_is_type(lambda_stop_early, bool)
        assert_is_type(glm_stop_early, bool)
        assert_is_type(glm_stop_early_error_fraction, float)
        assert_is_type(max_iter, int)
        assert_is_type(verbose, int)
        assert_is_type(family, str)
        assert family in ['logistic',
                          'elasticnet'], "family should be set to 'logistic' or 'elasticnet' but got " + family
        assert_is_type(lambda_max, float, None)
        assert_is_type(alpha_max, float, None)
        assert_is_type(alpha_min, float, None)

        if order is not None:
            assert_is_type(order, str)
            assert order in ['r', 'c'], "Order should be set to 'r' or 'c' but got " + order
            self.ord = ord(order)
        else:
            self.ord = None

        self.n = 0
        self.m_train = 0
        self.m_valid = 0
        self.source_dev = 0  # assume Dev=0 is source of data for _upload_data
        self.source_me = 0  # assume thread=0 is source of data for _upload_data
        if fit_intercept is True:
            self.fit_intercept = 1
        else:
            self.fit_intercept = 0
        self.lambda_min_ratio = lambda_min_ratio
        self.n_lambdas = n_lambdas
        self.n_folds = n_folds
        self.n_alphas = n_alphas
        self.uploaded_data = 0
        self.did_fit_ptr = 0
        self.did_predict = 0
        self.tol = tol
        if lambda_stop_early is True:
            self.lambda_stop_early = 1
        else:
            self.lambda_stop_early = 0
        if glm_stop_early is True:
            self.glm_stop_early = 1
        else:
            self.glm_stop_early = 0
        self.glm_stop_early_error_fraction = glm_stop_early_error_fraction
        self.max_iter = max_iter
        self.verbose = verbose
        self._family_str = family #Hold string value for family
        self._family = ord(family.split()[0][0])
        self.give_full_path = give_full_path
        if lambda_max is None:
            self.lambda_max = -1.0  # to trigger C code to compute
        else:
            self.lambda_max = lambda_max
        self.alpha_min = alpha_min  # as default
        self.alpha_max = alpha_max

        # Experimental features
        # TODO _shared_a and _standardize do not work currently. Always need to set to 0.
        self._shared_a = 0
        self._standardize = 0

        (self.n_gpus, devices) = device_count(n_gpus)

        if n_threads == None:
            # not required number of threads, but normal.  Bit more optimal to use 2 threads for CPU, but 1 thread per GPU is optimal.

            n_threads = (1 if self.n_gpus == 0 else self.n_gpus)
        self.n_threads = n_threads

        if not h2o4gpuGLMGPU:
            print(
                '\nWarning: Cannot create a H2O4GPU Elastic Net GPU Solver instance without linking Python module to a compiled H2O4GPU GPU library '
                '>Use CPU or add CUDA libraries to $PATH and re-run setup.py')

        if not h2o4gpuGLMCPU:
            print(
                '\nWarning: Cannot create a H2O4GPU Elastic Net CPU Solver instance without linking Python module to a compiled H2O4GPU '
                '>CPU library Use GPU or re-run setup.py')

        self.lib = None
        if self.n_gpus == 0 or h2o4gpuGLMGPU is None or devices == 0:
            if verbose > 0:
                print('Using CPU GLM solver %d %d' % (self.n_gpus, devices))
            self.lib = h2o4gpuGLMCPU
        elif self.n_gpus > 0 or h2o4gpuGLMGPU is None or devices == 0:
            if verbose > 0:
                print('Using GPU GLM solver with %d GPUs' % self.n_gpus)
            self.lib = h2o4gpuGLMGPU
        else:
            raise RuntimeError("Couldn't instantiate GLM Solver")




    def none_checks_simple(self, fail, give_full_path, verbose, order):

        # override self if passed parameter is not None
        if give_full_path is not None:
            self.give_full_path = give_full_path
        else:
            give_full_path = self.give_full_path

        if verbose is None:
            verbose = self.verbose

        if order in ['r', 'c']:
            self.ord = ord(order)
        elif order in [ord('r'), ord('c')]:
            self.ord = order
        elif self.ord is not None and fail:
            assert self.ord in ['r', 'c'], "Order should be set to 'r' or 'c' but got " + self.ord
        elif fail:
            assert False, "Order should be set to 'r' or 'c' or %d or %d but got " % (ord('r'), ord('c')) + order

        order = self.ord

        return give_full_path, verbose, order

    def none_checks(self, fail, give_full_path, tol, lambda_stop_early, glm_stop_early, glm_stop_early_error_fraction, max_iter, verbose, order):

        give_full_path, verbose, order = self.none_checks_simple(fail,give_full_path, verbose, order)

        if tol is not None:
            self.tol = tol
        else:
            tol = self.tol

        # Don't override self if pass option, but use self if option is None
        if lambda_stop_early is None:
            lambda_stop_early = self.lambda_stop_early
        if glm_stop_early is None:
            glm_stop_early = self.glm_stop_early
        if glm_stop_early_error_fraction is None:
            glm_stop_early_error_fraction = self.glm_stop_early_error_fraction
        if max_iter is None:
            max_iter = self.max_iter
        if verbose is None:
            verbose = self.verbose

        return give_full_path, tol, lambda_stop_early, glm_stop_early, glm_stop_early_error_fraction, max_iter, verbose, order


    # TODO Add typechecking
    # source_dev here because generally want to take in any pointer, not just from our test code
    """
    Train a model using GLM Solver with pointers to data on GPU

    :param source_dev
    :param m_train
    :parm n
    :param m_valid
    :param precision
    :param a
    :param b
    :param c
    :param d
    :param e
    :param give_full_path
    :param do_predict
    :param free_input_data
    :param tol
    :param lambda_stop_early
    :param glm_stop_early
    :param glm_stop_early_error_fraction
    :param max_iter
    :param verbose
    """

    def fit_ptr(
            self,
            source_dev,
            m_train,
            n,
            m_valid,
            precision,
            order,
            a, # trainX_ptr or train_xptr
            b, # trainY_ptr
            c, # validX_ptr
            d, # validY_ptr or valid_xptr  # keep consistent with later uses
            e, # weight_ptr
            give_full_path=None,
            do_predict=0,
            free_input_data=0,
            tol=None,
            lambda_stop_early=None,
            glm_stop_early=None,
            glm_stop_early_error_fraction=None,
            max_iter=None,
            verbose=None
    ):

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

        give_full_path, tol, lambda_stop_early, glm_stop_early, glm_stop_early_error_fraction, max_iter, verbose, order = self.none_checks(True, give_full_path, tol, lambda_stop_early, glm_stop_early, glm_stop_early_error_fraction, max_iter, verbose, order)

        # ###########

        if do_predict == 0 and self.did_fit_ptr == 1:
            self.free_sols()
        else:

            # otherwise don't clear solution, just use it

            pass

        # ###############

        self.did_fit_ptr = 1

        # ##############
        # not calling with self.source_dev because want option to never use default but instead input pointers from foreign code's pointers

        if hasattr(self, 'double_precision'):
            which_precision = self.double_precision
        else:
            which_precision = precision
            self.double_precision = precision

        # #############

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

        # ###############
        #

        c_size_t_p = POINTER(c_size_t)
        if which_precision == 1:
            self.mydtype = np.double
            self.myctype = c_double
            if verbose > 0:
                print('double precision fit')
                sys.stdout.flush()
            self.lib.elastic_net_ptr_double(
                c_int(self._family),
                c_int(do_predict),
                c_int(source_dev),
                c_int(1),
                c_int(self._shared_a),
                c_int(self.n_threads),
                c_int(self.n_gpus),
                c_int(self.ord),
                c_size_t(m_train),
                c_size_t(n),
                c_size_t(m_valid),
                c_int(self.fit_intercept),
                c_int(self._standardize),
                c_double(self.lambda_max),
                c_double(self.lambda_min_ratio),
                c_int(self.n_lambdas),
                c_int(self.n_folds),
                c_int(self.n_alphas),
                c_double(self.alpha_min),
                c_double(self.alpha_max),
                c_double(self.tol),
                c_int(lambda_stop_early),
                c_int(glm_stop_early),
                c_double(glm_stop_early_error_fraction),
                c_int(max_iter),
                c_int(verbose),
                a,
                b,
                c,
                d,
                e,
                give_full_path,
                pointer(x_vs_alpha_lambda),
                pointer(x_vs_alpha),
                pointer(valid_pred_vs_alpha_lambda),
                pointer(valid_pred_vs_alpha),
                cast(addressof(count_full), c_size_t_p),
                cast(addressof(count_short), c_size_t_p),
                cast(addressof(count_more), c_size_t_p),
            )
        else:
            self.mydtype = np.float
            self.myctype = c_float
            if verbose > 0:
                print('single precision fit')
                sys.stdout.flush()
            self.lib.elastic_net_ptr_float(
                c_int(self._family),
                c_int(do_predict),
                c_int(source_dev),
                c_int(1),
                c_int(self._shared_a),
                c_int(self.n_threads),
                c_int(self.n_gpus),
                c_int(self.ord),
                c_size_t(m_train),
                c_size_t(n),
                c_size_t(m_valid),
                c_int(self.fit_intercept),
                c_int(self._standardize),
                c_double(self.lambda_max),
                c_double(self.lambda_min_ratio),
                c_int(self.n_lambdas),
                c_int(self.n_folds),
                c_int(self.n_alphas),
                c_double(self.alpha_min),
                c_double(self.alpha_max),
                c_double(self.tol),
                c_int(lambda_stop_early),
                c_int(glm_stop_early),
                c_double(glm_stop_early_error_fraction),
                c_int(max_iter),
                c_int(verbose),
                a,
                b,
                c,
                d,
                e,
                give_full_path,
                pointer(x_vs_alpha_lambda),
                pointer(x_vs_alpha),
                pointer(valid_pred_vs_alpha_lambda),
                pointer(valid_pred_vs_alpha),
                cast(addressof(count_full), c_size_t_p),
                cast(addressof(count_short), c_size_t_p),
                cast(addressof(count_more), c_size_t_p),
            )

        #
        # if should or user wanted to save or free data, do that now that we are done using a,b,c,d,e
        # This means have to _upload_data() again before fit_ptr or predict_ptr or only call fit and predict

        if free_input_data == 1:
            self.free_data()

        # ####################################
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
        # #####

        if give_full_path == 1:
            num_all = int(count_full_value / (self.n_alphas
                                              * self.n_lambdas))
        else:
            num_all = int(count_short_value / self.n_alphas)

        #

        NUMALLOTHER = num_all - n
        NUMERROR = 3  # should be consistent with src/common/elastic_net_ptr.cpp
        NUMOTHER = NUMALLOTHER - NUMERROR
        if NUMOTHER != 3:
            print('NUMOTHER=%d but expected 3' % NUMOTHER)
            print('count_full_value=%d count_short_value=%d count_more_value=%d num_all=%d NUMALLOTHER=%d' %
                  (int(count_full_value), int(count_short_value),
                   int(count_more_value), int(num_all),
                   int(NUMALLOTHER)))
            sys.stdout.flush()
            exit(0)

        if give_full_path == 1 and do_predict == 0:
            # x_vs_alpha_lambda contains solution (and other data) for all lambda and alpha

            self.x_vs_alpha_lambdanew = \
                np.fromiter(cast(x_vs_alpha_lambda,
                                 POINTER(self.myctype)), dtype=self.mydtype,
                            count=count_full_value)
            self.x_vs_alpha_lambdanew = \
                np.reshape(self.x_vs_alpha_lambdanew, (self.n_lambdas,
                                                       self.n_alphas, num_all))
            self.x_vs_alpha_lambdapure = self.x_vs_alpha_lambdanew[:, :
            , 0:n]
            self.error_vs_alpha_lambda = self.x_vs_alpha_lambdanew[:, :
            , n:n + NUMERROR]
            self._lambdas = self.x_vs_alpha_lambdanew[:, :, n
                                                            + NUMERROR:n + NUMERROR + 1]
            self._alphas = self.x_vs_alpha_lambdanew[:, :, n + NUMERROR
                                                           + 1:n + NUMERROR + 2]
            self._tols = self.x_vs_alpha_lambdanew[:, :, n + NUMERROR
                                                         + 2:n + NUMERROR + 3]

            #

            self.solution.x_vs_alpha_lambdapure = \
                self.x_vs_alpha_lambdapure
            self.info.error_vs_alpha_lambda = self.error_vs_alpha_lambda
            self.info.lambdas = self._lambdas
            self.info.alphas = self._alphas
            self.info.tols = self._tols

            #

        if give_full_path == 1 and do_predict == 1:
            thecount = int(count_full_value / (n + NUMALLOTHER)
                           * m_valid)
            self.valid_pred_vs_alpha_lambdanew = \
                np.fromiter(cast(valid_pred_vs_alpha_lambda,
                                 POINTER(self.myctype)), dtype=self.mydtype,
                            count=thecount)
            self.valid_pred_vs_alpha_lambdanew = \
                np.reshape(self.valid_pred_vs_alpha_lambdanew,
                           (self.n_lambdas, self.n_alphas, m_valid))
            self.valid_pred_vs_alpha_lambdapure = \
                self.valid_pred_vs_alpha_lambdanew[:, :, 0:m_valid]

            #

        if do_predict == 0:  # give_full_path==0 or 1

            # x_vs_alpha contains only best of all lambda for each alpha

            self.x_vs_alphanew = np.fromiter(cast(x_vs_alpha,
                                                  POINTER(self.myctype)), dtype=self.mydtype,
                                             count=count_short_value)
            self.x_vs_alphanew = np.reshape(self.x_vs_alphanew,
                                            (self.n_alphas, num_all))
            self.x_vs_alphapure = self.x_vs_alphanew[:, 0:n]
            self.error_vs_alpha = self.x_vs_alphanew[:, n:n + NUMERROR]
            self._lambdas2 = self.x_vs_alphanew[:, n + NUMERROR:n
                                                                + NUMERROR + 1]
            self._alphas2 = self.x_vs_alphanew[:, n + NUMERROR + 1:n
                                                                   + NUMERROR + 2]
            self._tols2 = self.x_vs_alphanew[:, n + NUMERROR + 2:n
                                                                 + NUMERROR + 3]

            #

            self.solution.x_vs_alphapure = self.x_vs_alphapure
            self.info.error_vs_alpha = self.error_vs_alpha
            self.info.lambdas2 = self._lambdas2
            self.info.alphas2 = self._alphas2
            self.info.tols2 = self._tols2

        #

        if give_full_path == 0 and do_predict == 1:  # preds exclusively operate for x_vs_alpha or x_vs_alpha_lambda
            thecount = int(count_short_value / (n + NUMALLOTHER)
                           * m_valid)
            if verbose > 0:
                print('thecount=%d count_full_value=%d count_short_value=%d n=%d NUMALLOTHER=%d m_valid=%d'
                      % (
                          thecount,
                          count_full_value,
                          count_short_value,
                          n,
                          NUMALLOTHER,
                          m_valid,
                      ))
                sys.stdout.flush()
            self.valid_pred_vs_alphanew = \
                np.fromiter(cast(valid_pred_vs_alpha,
                                 POINTER(self.myctype)), dtype=self.mydtype,
                            count=thecount)
            self.valid_pred_vs_alphanew = \
                np.reshape(self.valid_pred_vs_alphanew, (self.n_alphas,
                                                         m_valid))
            self.valid_pred_vs_alphapure = self.valid_pred_vs_alphanew[:
            , 0:m_valid]

        return self

    # TODO Add typechecking

    """
    Train the model using GLM Solver

    :param train_x
    :param train_ y
    :param valid_x
    :param valid_y
    :param weight
    :param give_full_path
    :param do_predict
    :param free_input_data
    :param tol
    :param lambda_stop_early
    :param glm_stop_early
    :param glm_stop_early_error_fraction
    :param max_iter
    :param verbose
    """

    def fit(
            self,
            train_x,
            train_y,
            valid_x=None,
            valid_y=None,
            weight=None,
            give_full_path=None,
            do_predict=0,
            free_input_data=1,
            tol=None,
            lambda_stop_early=None,
            glm_stop_early=None,
            glm_stop_early_error_fraction=None,
            max_iter=None,
            verbose=None,
            order=None,
    ):

        give_full_path, tol, lambda_stop_early, glm_stop_early, glm_stop_early_error_fraction, max_iter, verbose, order = self.none_checks(False,
            give_full_path, tol, lambda_stop_early, glm_stop_early, glm_stop_early_error_fraction, max_iter,
            verbose, order)

        #If fit_intercept is set to True, then append to train_x array and valid_x array(if available)
        if self.fit_intercept:
            train_x = np.hstack([train_x, np.ones((train_x.shape[0], 1), dtype=train_x.dtype)])
            if valid_x is not None:
                valid_x = np.hstack([valid_x, np.ones((valid_x.shape[0], 1), dtype=valid_x.dtype)])

        ##############

        train_x_np, m_train, n1, fortran1 = _get_data(train_x, verbose=verbose)
        train_y_np, m_y, n_y, fortran2 = _get_data(train_y, verbose=verbose)
        valid_x_np, m_valid, n2, fortran3 = _get_data(valid_x, verbose=verbose)
        valid_y_np, m_valid_y, n_valid_y, fortran4 = _get_data(valid_y, verbose=verbose)
        weight_np, m_weight, n_weight, fortran5 = _get_data(weight, verbose=verbose)

        # check that inputs all have same 'c' or 'r' order
        fortran_list = [fortran1, fortran2, fortran3, fortran4, fortran5]
        _check_equal(fortran_list)

        # set order
        if order is None:
            if fortran1:
                order = 'c'
            else:
                order = 'r'
            self.ord = ord(order)

        # now can do checks

        # ###############
        # check do_predict input

        if m_train>=1 and m_y>=1 and m_train != m_y:
            print('training X and Y must have same number of rows, but m_train=%d m_y=%d\n' % (m_train, m_y))

        if do_predict == 0:
            if verbose > 0:
                if n1 >= 0 and m_y >= 0:
                    print('Correct train inputs')
                else:
                    print('Incorrect train inputs')
                    exit(0)
        if do_predict == 1:
            if n1 == -1 and n2 >= 0:
                if verbose > 0:
                    print('Correct prediction inputs')
            else:
                print('Incorrect prediction inputs: %d %d %d %d' % (n1,n2,m_valid_y,m_y))

        # ################

        if do_predict == 0:
            if n1 >= 0 and n2 >= 0 and n1 != n2:
                print('train_x and valid_x must have same number of columns, but n=%d n2=%d\n'
                      % (n1, n2))
                exit(0)
            else:
                n = n1  # either
        else:
            n = n2  # pick valid_x

        # #################

        if do_predict == 0:
            if m_valid >= 0 and m_valid_y >= 0 and m_valid != m_valid_y:
                print('valid_x and valid_y must have same number of rows, but m_valid=%d m_valid_y=%d\n' \
                      % (m_valid, m_valid_y))
                exit(0)
        else:
            # otherwise m_valid is used, and m_valid_y can be there or not (sets whether do error or not)
            pass

        # ################

        if do_predict == 0:
            if (m_valid == 0 or m_valid == -1) and n2 > 0 or m_valid \
                    > 0 and (n2 == 0 or n2 == -1):
                # if ((valid_x is not None and valid_y == None) or (valid_x == None and valid_y is not None)):

                print(
                    'Must input both valid_x and valid_y or neither.')  # TODO FIXME: Don't need valid_y if just want preds and no error, but don't return error in fit, so leave for now
                exit(1)

                #
        # #############

        source_dev = 0  # assume GPU=0 is fine as source
        (a, b, c, d, e) = self._upload_data(
            source_dev,
            train_x_np,
            train_y_np,
            valid_x_np,
            valid_y_np,
            weight_np,
        )
        precision = 0  # won't be used
        self.fit_ptr(
            source_dev,
            m_train,
            n,
            m_valid,
            precision,
            self.ord,
            a,
            b,
            c,
            d,
            e,
            give_full_path,
            do_predict=do_predict,
            free_input_data=free_input_data,
            tol=tol,
            lambda_stop_early=lambda_stop_early,
            glm_stop_early=glm_stop_early,
            glm_stop_early_error_fraction=glm_stop_early_error_fraction,
            max_iter=max_iter,
            verbose=verbose
        )
        return self

    # TODO Add typechecking
    """
    Obtains predictions from fitted GLM Solver
    :param valid_x
    :param valid_ y
    :param testweight
    :param give_full_path
    :param free_input_data
    """

    def predict(
            self,
            valid_x,
            valid_y=None,
            weight=None,
            give_full_path=None,
            free_input_data=1,
            verbose=0,
            order=None
    ):

        give_full_path, verbose, order = self.none_checks_simple(False, give_full_path, verbose, order)

        # if pass None train_x and train_y, then do predict using valid_x and weight (if given)
        # unlike _upload_data and fit_ptr (and so fit) don't free-up predictions since for single model might request multiple predictions.  User has to call finish themselves to cleanup.

        valid_x_np, n2, m_valid, fortran1 = _get_data(valid_x, verbose=self.verbose)
        valid_y_np, n_valid_y, m_valid_y, fortran2 = _get_data(valid_y, verbose=self.verbose)
        weight_np, n_weight, m_weight, fortran3 = _get_data(weight, verbose=self.verbose)

        # check that inputs all have same 'c' or 'r' order
        fortran_list = [fortran1, fortran2, fortran3]
        _check_equal(fortran_list)

        # override order
        if fortran1:
            order = 'c'
        else:
            order = 'r'
        self.ord = ord(order)

        ################
        # do checks on inputs

        do_predict = 1
        if give_full_path == 1:
            self.prediction_full = self.fit(
                None,
                None,
                valid_x_np,
                valid_y_np,
                weight_np,
                give_full_path,
                do_predict,
                free_input_data,
            ).valid_pred_vs_alpha_lambdapure
        else:
            self.prediction_full = None
        oldgivefullpath = self.give_full_path
        tempgivefullpath = 0
        self.prediction = self.fit(
            None,
            None,
            valid_x_np,
            valid_y_np,
            weight_np,
            tempgivefullpath,
            do_predict,
            free_input_data,
        ).valid_pred_vs_alphapure
        self.give_full_path = oldgivefullpath
        if give_full_path == 1:
            return self.prediction_full  # something like valid_y
        else:
            return self.prediction  # something like valid_y

    # TODO Add typechecking
    """
    Obtains predictions from fitted GLM Solver with data that is on GPU
    :param valid_xptr
    :param valid_ yptr
    :param free_input_data
    :param verbose
    """

    def predict_ptr(
            self,
            valid_xptr,
            valid_yptr=None,
            give_full_path=None,
            free_input_data=0,
            verbose=0,
            order=None
    ):
        # assume self.ord already set by fit_ptr() at least
        # override self if chose to pass this option

        give_full_path, verbose, order = self.none_checks_simple(True, give_full_path, verbose, order)

        do_predict = 1

        # print("%d %d %d %d %d" % (self.source_dev, self.m_train, self.n, self.m_valid, self.precision)) ; sys.stdout.flush()

        self.prediction = self.fit_ptr(
            self.source_dev,
            self.m_train,
            self.n,
            self.m_valid,
            self.precision,
            self.ord,
            self.a,
            self.b,
            valid_xptr,
            valid_yptr,
            self.e,
            0,
            do_predict,
            free_input_data,
            verbose,
        ).valid_pred_vs_alphapure
        if give_full_path == 1:  # then need to run twice
            self.prediction_full = self.fit_ptr(
                self.source_dev,
                self.m_train,
                self.n,
                self.m_valid,
                self.precision,
                self.ord,
                self.a,
                self.b,
                valid_xptr,
                valid_yptr,
                self.e,
                give_full_path,
                do_predict,
                free_input_data,
                verbose,
            ).valid_pred_vs_alpha_lambdapure
        else:
            self.prediction_full = None
        if give_full_path == 1:
            return self.prediction_full  # something like valid_y
        else:
            return self.prediction  # something like valid_y

    # TODO Add typechecking

    """
    Train a model using GLM Solver and predict on validation set
    :param train_x
    :param train_ y
    :param valid_x
    :param valid_y
    :param weight
    :param give_full_path
    :param free_input_data
    :param tol
    :param lambda_stop_early
    :param glm_stop_early
    :param glm_stop_early_error_fraction
    :param max_iter
    :param verbose
    """


    def fit_predict(
            self,
            train_x,
            train_y,
            valid_x=None,
            valid_y=None,
            weight=None,
            give_full_path=None,
            free_input_data=1,
            tol=None,
            lambda_stop_early=None,
            glm_stop_early=None,
            glm_stop_early_error_fraction=None,
            max_iter=None,
            verbose=None,
            order=None
    ):

        give_full_path, tol, lambda_stop_early, glm_stop_early, glm_stop_early_error_fraction, max_iter, verbose, order = self.none_checks(False, give_full_path, tol, lambda_stop_early, glm_stop_early, glm_stop_early_error_fraction, max_iter, verbose, order)

        do_predict = 0  # only fit at first

        # let fit() check and convert (to numpy) train_x, train_y, valid_x, valid_y, weight
        self.fit(
            train_x,
            train_y,
            valid_x,
            valid_y,
            weight,
            give_full_path,
            do_predict,
            free_input_data=0,
            tol=tol,
            lambda_stop_early=lambda_stop_early,
            glm_stop_early=glm_stop_early,
            glm_stop_early_error_fraction=glm_stop_early_error_fraction,
            max_iter=max_iter,
            verbose=verbose,
        )
        if valid_x == None:
            if give_full_path == 1:
                self.prediction_full = self.predict(train_x, train_y,
                                                    weight=weight,
                                                    give_full_path=give_full_path,
                                                    free_input_data=free_input_data)
            else:
                self.prediction_full = None
            self.prediction = self.predict(train_x, train_y,
                                           weight=weight, give_full_path=0,
                                           free_input_data=free_input_data)
        else:
            if give_full_path == 1:
                self.prediction_full = self.predict(valid_x, valid_y,
                                                    weight=weight,
                                                    give_full_path=give_full_path,
                                                    free_input_data=free_input_data)
            else:
                self.prediction_full = None
            self.prediction = self.predict(valid_x, valid_y,
                                           weight=weight, give_full_path=0,
                                           free_input_data=free_input_data)
        if give_full_path:
            return self.prediction_full  # something like valid_y
        else:
            return self.prediction  # something like valid_y

    # TODO Add typechecking
    """
     Train a model using GLM Solver with pointers to data on GPU and predict on validation set that also has a pointer on GPU
    :param source_dev
    :param m_train
    :parm n
    :param m_valid
    :param precision
    :param a
    :param b
    :param c
    :param d
    :param e
    :param give_full_path
    :param free_input_data
    :param tol
    :param lambda_stop_early
    :param glm_stop_early
    :param glm_stop_early_error_fraction
    :param max_iter
    :param verbose
    """

    def fit_predict_ptr(
            self,
            source_dev,
            m_train,
            n,
            m_valid,
            precision,
            order,
            a,
            b,
            c,
            d,
            e,
            give_full_path=None,
            free_input_data=0,
            tol=None,
            lambda_stop_early=None,
            glm_stop_early=None,
            glm_stop_early_error_fraction=None,
            max_iter=None,
            verbose=None,
    ):
        give_full_path, tol, lambda_stop_early, glm_stop_early, glm_stop_early_error_fraction, max_iter, verbose, order = self.none_checks(True, give_full_path, tol, lambda_stop_early, glm_stop_early, glm_stop_early_error_fraction, max_iter, verbose, order)

        do_predict = 0  # only fit at first

        self.fit_ptr(
            source_dev,
            m_train,
            n,
            m_valid,
            precision,
            self.ord,
            a,
            b,
            c,
            d,
            e,
            give_full_path,
            do_predict,
            free_input_data=0,
            tol=tol,
            lambda_stop_early=lambda_stop_early,
            glm_stop_early=glm_stop_early,
            glm_stop_early_error_fraction=glm_stop_early_error_fraction,
            max_iter=max_iter,
            verbose=verbose,
        )
        if c is None or c is c_void_p(0):
            self.prediction = self.predict_ptr(a, b, 0,
                                               free_input_data=free_input_data)
            if give_full_path == 1:
                self.prediction_full = self.predict_ptr(a, b,
                                                        give_full_path, free_input_data=free_input_data)
        else:
            self.prediction = self.predict_ptr(c, d, 0,
                                               free_input_data=free_input_data)
            if give_full_path == 1:
                self.prediction_full = self.predict_ptr(c, d,
                                                        give_full_path, free_input_data=free_input_data)
        if give_full_path:
            return self.prediction_full  # something like valid_y
        else:
            return self.prediction  # something like valid_y

    def fit_transform(
            self,
            train_x,
            train_y,
            valid_x=None,
            valid_y=None,
            weight=None,
            give_full_path=None,
            free_input_data=1,
            tol=None,
            lambda_stop_early=None,
            glm_stop_early=None,
            glm_stop_early_error_fraction=None,
            max_iter=None,
            verbose=None,
    ):

        return (self.fit_predict(self, train_x, train_y, valid_x, valid_y, weight, give_full_path, free_input_data,
                                 tol, lambda_stop_early, glm_stop_early, glm_stop_early_error_fraction, max_iter,
                                 verbose))

    def transform(self):
        return

    # ################### Properties and setters of properties

    @property
    def family(self):
        return self._family_str

    @family.setter
    def family(self, value):

        # add check

        self.family = value

    @property
    def shared_a(self):
        return self._shared_a

    @shared_a.setter
    def shared_a(self, value):

        # add check

        self.__shared_a = value

    @property
    def standardize(self):
        return self._standardize

    @standardize.setter
    def standardize(self, value):

        # add check

        self._standardize = value

    @property
    def X(self):
        if self.give_full_path == 1:
            return self.x_vs_alpha_lambdapure
        else:
            return self.x_vs_alphapure

    @property
    def X_full(self):
        return self.x_vs_alpha_lambdapure

    @property
    def X_best(self):
        return self.x_vs_alphapure

    @property
    def validPreds(self):
        if self.give_full_path == 1:
            return self.valid_pred_vs_alpha_lambdapure
        else:
            return self.valid_pred_vs_alphapure

    @property
    def validPreds_full(self):
        return self.valid_pred_vs_alpha_lambdapure

    @property
    def validPreds_best(self):
        return self.valid_pred_vs_alphapure

    @property
    def error(self):
        if self.give_full_path == 1:
            return self.error_vs_alpha_lambda
        else:
            return self.error_vs_alpha

    @property
    def lambdas(self):
        if self.give_full_path == 1:
            return self._lambdas
        else:
            return self._lambdas2

    @lambdas.setter
    def lambdas(self, value):

        # add check

        self._lambdas = value

    # @lambdas2.setter
    # def lambdas2(self, value):
    #    # add check
    #    self._lambdas2 = value

    @property
    def alphas(self):
        if self.give_full_path == 1:
            return self._alphas
        else:
            return self._alphas2

    @alphas.setter
    def alphas(self, value):
        self._alphas = value

    @property
    def tols(self):
        if self.give_full_path == 1:
            return self._tols
        else:
            return self._tols2

    @tols.setter
    def tols(self, value):
        self._tols = value

    @property
    def error_full(self):
        return self.error_vs_alpha_lambda

    @property
    def lambdas_full(self):
        return self._lambdas

    @property
    def alphas_full(self):
        return self._alphas

    @property
    def tols_full(self):
        return self._tols

    @property
    def error_best(self):
        return self.error_vs_alpha

    @property
    def lambdas_best(self):
        return self._lambdas2

    @property
    def alphas_best(self):
        return self._alphas2

    @property
    def tols_best(self):
        return self._tols2

    # ################### Free up memory functions

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

    # TODO(jon): add option to pass in min max of alphas and lambdamax.

    # Util/Hidden Functions

    def _upload_data(
            self,
            source_dev,
            train_x,
            train_y,
            valid_x=None,
            valid_y=None,
            weight=None,
    ):
        if self.uploaded_data == 1:
            self.free_data()
        self.uploaded_data = 1

        #
        # ################

        self.double_precision1, m_train, n1 = _check_data_size(train_x, verbose=self.verbose)
        self.m_train = m_train
        self.double_precision3, m_train2, n_train2 = _check_data_size(train_y, verbose=self.verbose)
        self.double_precision2, m_valid, n2 = _check_data_size(valid_x, verbose=self.verbose)
        self.m_valid = m_valid
        self.double_precision4, m_valid2, n_valid2 = _check_data_size(valid_y, verbose=self.verbose)
        self.double_precision5, m_train3, n_train3 = _check_data_size(weight, verbose=self.verbose)


        if self.double_precision1 >= 0 and self.double_precision2 >= 0:
            if self.double_precision1 != self.double_precision2:
                print('train_x and valid_x must be same precision')
                exit(0)
            else:
                self.double_precision = self.double_precision1  # either one
        elif self.double_precision1 >= 0:
            self.double_precision = self.double_precision1
        elif self.double_precision2 >= 0:
            self.double_precision = self.double_precision2

        # ##############

        if self.double_precision1 >= 0 and self.double_precision3 >= 0:
            if self.double_precision1 != self.double_precision3:
                print('train_x and train_y must be same precision')
                exit(0)

        # ##############

        if self.double_precision2 >= 0 and self.double_precision4 >= 0:
            if self.double_precision2 != self.double_precision4:
                print('valid_x and valid_y must be same precision')
                exit(0)

        # ##############

        if self.double_precision3 >= 0 and self.double_precision5 >= 0:
            if self.double_precision3 != self.double_precision5:
                print('train_y and weight must be same precision')
                exit(0)

        # ##############

        n = -1
        if n1 >= 0 and n2 >= 0:
            if n1 != n2:
                print('train_x and valid_x must have same number of columns')
                exit(0)
            else:
                n = n1  # either one
        elif n1 >= 0:
            n = n1
        elif n2 >= 0:
            n = n2
        self.n = n

        # ###############

        a = c_void_p(0)
        b = c_void_p(0)
        c = c_void_p(0)
        d = c_void_p(0)
        e = c_void_p(0)
        if self.double_precision == 1:
            c_ftype = c_double

            if self.verbose > 0:
                print('Detected np.float64')
                sys.stdout.flush()
        else:
            c_ftype = c_float

            if self.verbose > 0:
                print('Detected np.float32')
                sys.stdout.flush()

        A = _convert_to_ptr(train_x, c_ftype)
        B = _convert_to_ptr(train_y, c_ftype)
        C = _convert_to_ptr(valid_x, c_ftype)
        D = _convert_to_ptr(valid_y, c_ftype)
        E = _convert_to_ptr(weight, c_ftype)

        if self.double_precision == 1:
            status = self.lib.make_ptr_double(
                c_int(self._shared_a),
                c_int(self.source_me),
                c_int(source_dev),
                c_size_t(m_train),
                c_size_t(n),
                c_size_t(m_valid),
                c_int(self.ord),
                A,
                B,
                C,
                D,
                E,
                pointer(a),
                pointer(b),
                pointer(c),
                pointer(d),
                pointer(e),
            )
        elif self.double_precision == 0:
            status = self.lib.make_ptr_float(
                c_int(self._shared_a),
                c_int(self.source_me),
                c_int(source_dev),
                c_size_t(m_train),
                c_size_t(n),
                c_size_t(m_valid),
                c_int(self.ord),
                A,
                B,
                C,
                D,
                E,
                pointer(a),
                pointer(b),
                pointer(c),
                pointer(d),
                pointer(e),
            )
        else:
            print('Unknown numpy type detected')
            print(train_x.dtype)
            sys.stdout.flush()
            return (a, b, c, d, e)

        assert status == 0, 'Failure uploading the data'

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
        return (a, b, c, d, e)

    def summary(self):
        """
        Obtain model summary, which is error per alpha across train, cv, and validation

        Error is logloss for classification and RMSE(Root Mean Squared Error) for regression.
        """
        error_train = pd.DataFrame(self.error_best, index=self.alphas)
        if self.family == "logistic":
            print("Logloss per alpha value (-1.00 = missing)\n")
        else:
            print("RMSE per alpha value (-1.00 = missing)\n")
        headers = ["Alphas", "Train", "CV", "Valid"]
        print(tabulate(error_train, headers=headers, tablefmt="pipe", floatfmt=".2f"))