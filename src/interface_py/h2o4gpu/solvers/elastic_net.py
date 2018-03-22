#- * - encoding : utf - 8 - * -
# pylint: disable=fixme, line-too-long
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import sys
import time
from ctypes import c_int, c_float, c_double, c_void_p, c_size_t, POINTER, \
    pointer, cast, addressof
import warnings

import numpy as np
import pandas as pd
from tabulate import tabulate
from h2o4gpu.linear_model import coordinate_descent as sk
from ..solvers.utils import _setter

from ..libs.lib_elastic_net import GPUlib, CPUlib
from ..solvers.utils import prepare_and_upload_data, free_data, free_sols
from ..util.gpu import device_count


class ElasticNetH2O(object):
    """H2O Elastic Net Solver for GPUs

       Parameters
       ----------
       n_threads : int, (Default=None)
           Number of threads to use in the gpu.
           Each thread is an independent model builder.

       gpu_id : int, optional, (default=0)
           ID of the GPU on which the algorithm should run.

       n_gpus : int, (Default=-1)
           Number of gpu's to use in GLM solver.

       order : string, (Default='r')
           Row or Column major for C/C++ backend. Default is 'r'.
           Must be 'r' (Row major) or 'c' (Column major).

       fit_intercept : bool, (default=True)
           Include constant term in the model.

       lambda_min_ratio: float, (Default=1E-7).
           Minimum lambda ratio to maximum lambda, used
           in lambda search.

       n_lambdas : int, (Default=100)
           Number of lambdas to be used in a search.

       n_folds : int,  (Default=1)
           Number of cross validation folds.

       n_alphas : int, (Default=5)
           Number of alphas to be used in a search.

       tol : float, (Default=1E-2)
           Relative tolerance.

       tol_seek_factor : float, (Default=1E-1)
           Factor of tolerance to seek
           once below null model accuracy.  Default is 1E-1, so seeks tolerance
           of 1E-3 once below null model accuracy for tol=1E-2.

       lambda_stop_early : float, (Default=True)
           Stop early when there is no more relative
           improvement on train or validation.

       glm_stop_early : bool, (Default=True)
           Stop early when there is no more relative
           improvement in the primary and dual residuals for ADMM.

       glm_stop_early_error_fraction : float, (Default=1.0)
           Relative tolerance for metric-based stopping criterion (stop if relative improvement is not at
           least this much).

       max_iter : int, (Default=5000)
           Maximum number of iterations.

       verbose : int, (Default=0)
           Print verbose information to the console if set to > 0.

       family : string, (Default="elasticnet")
           "logistic" for classification with logistic regression.
           Defaults to "elasticnet" for regression.
           Must be "logistic" or "elasticnet".

       store_full_path: int, (Default=0)
           Whether to store full solution for all alphas
           and lambdas.  If 1, then during predict will compute best
           and full predictions.

       lambda_max : int, (Default=None)
           Maximum Lambda value to use.
           Default is None, and then internally compute standard maximum

       alpha_max : float, (Default=1.0)
           Maximum alpha.

       alpha_min : float, (Default=0.0)
           Minimum alpha.

       alphas: list, tuple, array, or numpy 1D array of alphas (Default=None)
           overrides n_alphas, alpha_min, and alpha_max.

       lambdas: list, tuple, array, or numpy 1D array of lambdas (Default=None)
           overrides n_lambdas, lambda_max, and lambda_min_ratio.

       double_precision: int, (Default=None)
           Internally set unless using _ptr methods. Value can either be
           0 (float32) or 1(float64)

       order : string, (Default=None)
           Order of data. Default is None, and internally
           determined (unless using _ptr methods) whether
           row 'r' or column 'c' major order.
       """

    class info:
        pass

    class solution:
        pass

    def __init__(self,
                 n_threads=None,
                 gpu_id=0,
                 n_gpus=-1,
                 fit_intercept=True,
                 lambda_min_ratio=1E-7,
                 n_lambdas=100,
                 n_folds=5,
                 n_alphas=5,
                 tol=1E-2,
                 tol_seek_factor=1E-1,
                 lambda_stop_early=True,
                 glm_stop_early=True,
                 glm_stop_early_error_fraction=1.0,
                 max_iter=5000,
                 verbose=0,
                 family='elasticnet',
                 store_full_path=0,
                 lambda_max=None,
                 alpha_max=1.0,
                 alpha_min=0.0,
                 alphas=None,
                 lambdas=None,
                 double_precision=None,
                 order=None):
        assert family in ['logistic',
                          'elasticnet'], \
            "family should be 'logistic' or 'elasticnet' but got " + family

        self.double_precision = double_precision

        if order is not None:
            assert order in ['r',
                             'c'], \
                "Order should be set to 'r' or 'c' but got " + order
            self.ord = ord(order)
        else:
            self.ord = None
        self.dtype = None

        ##############################
        #overrides of input parameters
        #override these if pass alphas or lambdas
        if alphas is not None:
            alphas = np.ascontiguousarray(np.asarray(alphas))
            n_alphas = np.shape(alphas)[0]
        if lambdas is not None:
            lambdas = np.ascontiguousarray(np.asarray(lambdas))
            n_lambdas = np.shape(lambdas)[0]

        ##############################
        #self assignments
        self.n = 0
        self.m_train = 0
        self.m_valid = 0
        self.source_dev = 0  # assume Dev=0 is source of data for upload_data
        self.source_me = 0  # assume thread=0 is source of data for upload_data
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
        self.tol_seek_factor = tol_seek_factor
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
        self._family_str = family  # Hold string value for family
        self._family = ord(family.split()[0][0])
        self.store_full_path = store_full_path
        if lambda_max is None:
            self.lambda_max = -1.0  # to trigger C code to compute
        else:
            self.lambda_max = lambda_max
        self.alpha_min = alpha_min  # as default
        self.alpha_max = alpha_max

        self.alphas_list = alphas
        self.lambdas_list = lambdas

        # default None for _full stuff
        self.error_vs_alpha_lambda = None
        self.intercept_ = None
        self._tols2 = None
        self._lambdas2 = None
        self._alphas2 = None
        self.error_vs_alpha = None
        self.valid_pred_vs_alphapure = None
        self.x_vs_alphapure = None
        self.x_vs_alpha_lambdanew = None
        self.x_vs_alpha_lambdapure = None
        self.valid_pred_vs_alpha_lambdapure = None
        self._lambdas = None
        self._alphas = None
        self._tols = None
        self.intercept2_ = None

        #Experimental features
        #TODO _shared_a and _standardize do not work currently.
        #TODO Always need to set to 0.
        self._shared_a = 0
        self._standardize = 0

        (self.n_gpus, devices) = device_count(n_gpus)
        gpu_id = gpu_id % devices
        self._gpu_id = gpu_id
        self._total_n_gpus = devices

        if n_threads is None:
            #Not required number of threads, but normal.
            #Bit more optimal to use 2 threads for CPU,
            #but 1 thread per GPU is optimal.
            n_threads = (1 if self.n_gpus == 0 else self.n_gpus)

        self.n_threads = n_threads

        gpu_lib = GPUlib().get()
        cpu_lib = CPUlib().get()

        if self.n_gpus == 0 or gpu_lib is None or devices == 0:
            if verbose > 0:
                print('Using CPU GLM solver %d %d' % (self.n_gpus, devices))
            self.lib = cpu_lib
        elif self.n_gpus > 0 or gpu_lib is None or devices == 0:
            if verbose > 0:
                print('Using GPU GLM solver with %d GPUs' % self.n_gpus)
            self.lib = gpu_lib
        else:
            raise RuntimeError("Couldn't instantiate GLM Solver")

        self.x_vs_alpha_lambda = None
        self.x_vs_alpha = None
        self.valid_pred_vs_alpha_lambda = None
        self.valid_pred_vs_alpha = None
        self.count_full = None
        self.count_short = None
        self.count_more = None

#TODO Add typechecking

    def fit(self,
            train_x=None,
            train_y=None,
            valid_x=None,
            valid_y=None,
            sample_weight=None,
            free_input_data=1):
        """Train a GLM

        :param ndarray train_x : Training features array

        :param ndarray train_ y : Training response array

        :param ndarray valid_x : Validation features

        :param ndarray valid_ y : Validation response

        :param ndarray weight : Observation weights

        :param int free_input_data : Indicate if input data should be freed
            at the end of fit(). Default is 1.
        """

        source_dev = 0
        if not (train_x is None and train_y is None and valid_x is None and
                valid_y is None and sample_weight is None):

            self.prepare_and_upload_data = prepare_and_upload_data(
                self,
                train_x=train_x,
                train_y=train_y,
                valid_x=valid_x,
                valid_y=valid_y,
                sample_weight=sample_weight,
                source_dev=source_dev)

        else:
            #if all None, just assume fitting with new parameters
            #and all else uses self.
            pass

        self.fit_ptr(
            self.m_train,
            self.n,
            self.m_valid,
            self.double_precision,
            self.ord,
            self.a,
            self.b,
            self.c,
            self.d,
            self.e,
            free_input_data=free_input_data,
            source_dev=source_dev)
        return self

#TODO Add typechecking
    def predict(self,
                valid_x=None,
                valid_y=None,
                sample_weight=None,
                free_input_data=1):
        """Predict on a fitted GLM and get back class predictions for binomial models
        for classification and predicted values for regression.

        :param ndarray valid_x : Validation features

        :param ndarray valid_y : Validation response

        :param ndarray weight : Observation weights

        :param int free_input_data : Indicate if input data should be freed at
            the end of fit(). Default is 1.
        """
        res = self.predict_proba(valid_x, valid_y, sample_weight, free_input_data)
        if self.family == "logistic":
            res[res < 0.5] = 0
            res[res > 0.5] = 1
        return res

    def predict_proba(self,
                      valid_x=None,
                      valid_y=None,
                      sample_weight=None,
                      free_input_data=1):
        """Predict on a fitted GLM and get back uncalibrated probabilities for classification models

        :param ndarray valid_x : Validation features

        :param ndarray valid_y : Validation response

        :param ndarray weight : Observation weights

        :param int free_input_data : Indicate if input data should be freed at
            the end of fit(). Default is 1.
        """

        source_dev = 0
        if not (valid_x is None and valid_y is None and sample_weight is None):

            prepare_and_upload_data(
                self,
                train_x=None,
                train_y=None,
                valid_x=valid_x,
                valid_y=valid_y,
                sample_weight=sample_weight,
                source_dev=source_dev)
        else:
            pass

#save global variable
        oldstorefullpath = self.store_full_path

        if self.store_full_path == 1:
            self.store_full_path = 1
            self._fitorpredict_ptr(
                source_dev,
                self.m_train,
                self.n,
                self.m_valid,
                self.double_precision,
                self.ord,
                self.a,
                self.b,
                self.c,
                self.d,
                self.e,
                do_predict=1,
                free_input_data=free_input_data)

        self.store_full_path = 0
        self._fitorpredict_ptr(
            source_dev,
            self.m_train,
            self.n,
            self.m_valid,
            self.double_precision,
            self.ord,
            self.a,
            self.b,
            self.c,
            self.d,
            self.e,
            do_predict=1,
            free_input_data=free_input_data)

        #restore variable
        self.store_full_path = oldstorefullpath
        return self.valid_pred_vs_alphapure  # something like valid_y
#TODO Add type checking
#source_dev here because generally want to take in any pointer,
#not just from our test code

    def fit_ptr(
            self,
            m_train,
            n,
            m_valid,
            double_precision,
            order,
            a,  # trainX_ptr or train_xptr
            b,  # trainY_ptr
            c,  # validX_ptr
            d,  # validY_ptr or valid_xptr  # keep consistent with later uses
            e,  # weight_ptr
            free_input_data=0,
            source_dev=0):
        """Train a GLM with pointers to data on the GPU
           (if fit_intercept, then you should have added 1's as
           last column to m_train)


        :param m_train Number of rows in the training set

        :param n Number of columns in the training set

        :param m_valid Number of rows in the validation set

        :param double_precision float32 (0) or double point precision (1) of fit
            No Default.

        :param order: Order of data.
            Default is None, and assumed set by constructor or upload_data
            whether row 'r' or column 'c' major order.

        :param a Pointer to training features array

        :param b Pointer to training response array

        :param c Pointer to validation features

        :param d Pointer to validation response

        :param e Pointer to weight column

        :param int free_input_data : Indicate if input data should be freed at
            the end of fit(). Default is 1.

        :param source_dev GPU ID of device
        """

        time_fit0 = time.time()

        self._fitorpredict_ptr(
            source_dev,
            m_train,
            n,
            m_valid,
            double_precision,
            order,
            a,
            b,
            c,
            d,
            e,
            do_predict=0,
            free_input_data=free_input_data)
        self.time_fitonly = time.time() - time_fit0

#TODO Add type checking

#source_dev here because generally want to take in any pointer,
#not just from our test code

    def _fitorpredict_ptr(
            self,
            source_dev,
            m_train,
            n,
            m_valid,
            double_precision,
            order,
            a,  # trainX_ptr or train_xptr
            b,  # trainY_ptr
            c,  # validX_ptr
            d,  # validY_ptr or valid_xptr  # keep consistent with later uses
            e,  # weight_ptr
            do_predict=0,
            free_input_data=0):
        """Train a GLM with pointers to data on the GPU
           (if fit_intercept, then you should have added 1's as
           last column to m_train)


        :param source_dev GPU ID of device

        :param m_train Number of rows in the training set

        :param n Number of columns in the training set

        :param m_valid Number of rows in the validation set

        :param double_precision float32 (0) or double point precision (1) of fit
            No Default.

        :param order: Order of data.  Default is None and set elsewhere
            whether row 'r' or column 'c' major order.

        :param a Pointer to training features array

        :param b Pointer to training response array

        :param c Pointer to validation features

        :param d Pointer to validation response

        :param e Pointer to weight column

        :param int do_predict : Indicate if prediction should be done on
            validation set after train. Default is 0.

        :param int free_input_data : Indicate if input data should be freed at
            the end of fit(). Default is 1.
        """

        #store some things for later call to predict_ptr()

        self.source_dev = source_dev
        self.m_train = m_train
        self.n = n
        self.m_valid = m_valid
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e

        # ########## #

        #if fitted earlier clear
        #otherwise don't clear solution, just use it
        if do_predict == 0 and self.did_fit_ptr == 1:
            free_sols(self)

# ############## #

        self.did_fit_ptr = 1

        # ##############
        #not calling with self.source_dev because want option to never use
        #default but instead input pointers from foreign code's pointers

        if order is not None:  # set order if not already set
            if order in ['r', 'c']:
                self.ord = ord(order)
            else:
                self.ord = order

        if hasattr(self,
                   'double_precision') and self.double_precision is not None:
            which_precision = self.double_precision
        else:
            which_precision = double_precision
            self.double_precision = double_precision

# ############ #

        if do_predict == 0:

            #initialize if doing fit

            x_vs_alpha_lambda = c_void_p(0)
            x_vs_alpha = c_void_p(0)
            valid_pred_vs_alpha_lambda = c_void_p(0)
            valid_pred_vs_alpha = c_void_p(0)
            count_full = c_size_t(0)
            count_short = c_size_t(0)
            count_more = c_size_t(0)
        else:

            #restore if predict

            x_vs_alpha_lambda = self.x_vs_alpha_lambda
            x_vs_alpha = self.x_vs_alpha
            valid_pred_vs_alpha_lambda = self.valid_pred_vs_alpha_lambda
            valid_pred_vs_alpha = self.valid_pred_vs_alpha
            count_full = self.count_full
            count_short = self.count_short
            count_more = self.count_more

# ############## #
#

        c_size_t_p = POINTER(c_size_t)
        if which_precision == 1:
            c_elastic_net = self.lib.elastic_net_ptr_double
            self.dtype = np.float64
            self.myctype = c_double
            if self.verbose > 0:
                print('double precision fit')
                sys.stdout.flush()
        else:
            c_elastic_net = self.lib.elastic_net_ptr_float
            self.dtype = np.float32
            self.myctype = c_float
            if self.verbose > 0:
                print('single precision fit')
                sys.stdout.flush()

#precision - independent commands
        if self.alphas_list is not None:
            pass_alphas = (self.alphas_list.astype(self.dtype, copy=False))
            c_alphas = pass_alphas.ctypes.data_as(POINTER(self.myctype))
        else:
            c_alphas = cast(0, POINTER(self.myctype))
        if self.lambdas_list is not None:
            pass_lambdas = (self.lambdas_list.astype(self.dtype, copy=False))
            c_lambdas = pass_lambdas.ctypes.data_as(POINTER(self.myctype))
        else:
            c_lambdas = cast(0, POINTER(self.myctype))

#call elastic net in C backend
        c_elastic_net(
            c_int(self._family),
            c_int(do_predict),
            c_int(source_dev),
            c_int(1),
            c_int(self._shared_a),
            c_int(self.n_threads),
            c_int(self._gpu_id),
            c_int(self.n_gpus),
            c_int(self._total_n_gpus),
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
            c_alphas,
            c_lambdas,
            c_double(self.tol),
            c_double(self.tol_seek_factor),
            c_int(self.lambda_stop_early),
            c_int(self.glm_stop_early),
            c_double(self.glm_stop_early_error_fraction),
            c_int(self.max_iter),
            c_int(self.verbose),
            a,
            b,
            c,
            d,
            e,
            self.store_full_path,
            pointer(x_vs_alpha_lambda),
            pointer(x_vs_alpha),
            pointer(valid_pred_vs_alpha_lambda),
            pointer(valid_pred_vs_alpha),
            cast(addressof(count_full), c_size_t_p),
            cast(addressof(count_short), c_size_t_p),
            cast(addressof(count_more), c_size_t_p),
        )
        #if should or user wanted to save or free data,
        #do that now that we are done using a, b, c, d, e
        #This means have to upload_data() again before fit_ptr
        # or predict_ptr or only call fit and predict

        if free_input_data == 1:
            free_data(self)

# ####################################
#PROCESS OUTPUT
#save pointers

        self.x_vs_alpha_lambda = x_vs_alpha_lambda
        self.x_vs_alpha = x_vs_alpha
        self.valid_pred_vs_alpha_lambda = valid_pred_vs_alpha_lambda
        self.valid_pred_vs_alpha = valid_pred_vs_alpha
        self.count_full = count_full
        self.count_short = count_short
        self.count_more = count_more

        count_full_value = count_full.value
        count_short_value = count_short.value
        count_more_value = count_more.value

        if self.store_full_path == 1:
            num_all = int(count_full_value / (self.n_alphas * self.n_lambdas))
        else:
            num_all = int(count_short_value / self.n_alphas)

        num_all_other = num_all - n
        num_error = 3  # should be consistent w/ src/common/elastic_net_ptr.cpp
        num_other = num_all_other - num_error
        if num_other != 3:
            print('num_other=%d but expected 3' % num_other)
            print('count_full_value=%d '
                  'count_short_value=%d '
                  'count_more_value=%d '
                  'num_all=%d num_all_other=%d' % (int(count_full_value),
                                                   int(count_short_value),
                                                   int(count_more_value),
                                                   int(num_all),
                                                   int(num_all_other)))
            sys.stdout.flush()
            #TODO raise an exception instead
            exit(0)

        if self.store_full_path == 1 and do_predict == 0:
            #x_vs_alpha_lambda contains solution(and other data)
            #for all lambda and alpha

            self.x_vs_alpha_lambdanew = \
                np.fromiter(cast(x_vs_alpha_lambda,
                                 POINTER(self.myctype)), dtype=self.dtype,
                            count=count_full_value)

            self.x_vs_alpha_lambdanew = \
                np.reshape(self.x_vs_alpha_lambdanew, (self.n_lambdas,
                                                       self.n_alphas, num_all))

            self.x_vs_alpha_lambdapure = \
                self.x_vs_alpha_lambdanew[:, :, 0:n]

            self.error_vs_alpha_lambda = \
                self.x_vs_alpha_lambdanew[:, :, n:n + num_error]

            self._lambdas = \
                self.x_vs_alpha_lambdanew[:, :, n + num_error:n + num_error + 1]

            self._alphas = self.x_vs_alpha_lambdanew[:, :, n + num_error + 1:
                                                     n + num_error + 2]

            self._tols = self.x_vs_alpha_lambdanew[:, :, n + num_error + 2:
                                                   n + num_error + 3]

            if self.fit_intercept == 1:
                self.intercept_ = self.x_vs_alpha_lambdapure[:, :, -1]
            else:
                self.intercept_ = None

        if self.store_full_path == 1 and do_predict == 1:
            thecount = int(count_full_value / (n + num_all_other) * m_valid)
            self.valid_pred_vs_alpha_lambdanew = \
                np.fromiter(cast(valid_pred_vs_alpha_lambda,
                                 POINTER(self.myctype)), dtype=self.dtype,
                            count=thecount)
            self.valid_pred_vs_alpha_lambdanew = \
                np.reshape(self.valid_pred_vs_alpha_lambdanew,
                           (self.n_lambdas, self.n_alphas, m_valid))
            self.valid_pred_vs_alpha_lambdapure = \
                self.valid_pred_vs_alpha_lambdanew[:, :, 0:m_valid]

        if do_predict == 0:  # store_full_path==0 or 1
            #x_vs_alpha contains only best of all lambda for each alpha

            self.x_vs_alphanew = np.fromiter(
                cast(x_vs_alpha, POINTER(self.myctype)),
                dtype=self.dtype,
                count=count_short_value)
            self.x_vs_alphanew = np.reshape(self.x_vs_alphanew,
                                            (self.n_alphas, num_all))
            self.x_vs_alphapure = self.x_vs_alphanew[:, 0:n]
            self.error_vs_alpha = self.x_vs_alphanew[:, n:n + num_error]
            self._lambdas2 = self.x_vs_alphanew[:, n + num_error:
                                                n + num_error + 1]
            self._alphas2 = self.x_vs_alphanew[:, n + num_error + 1:
                                               n + num_error + 2]
            self._tols2 = self.x_vs_alphanew[:, n + num_error + 2:
                                             n + num_error + 3]

            if self.fit_intercept == 1:
                self.intercept2_ = self.x_vs_alphapure[:, -1]
            else:
                self.intercept2_ = None

#preds exclusively operate for x_vs_alpha or x_vs_alpha_lambda
        if self.store_full_path == 0 and do_predict == 1:
            thecount = int(count_short_value / (n + num_all_other) * m_valid)
            if self.verbose > 0:
                print('thecount=%d '
                      'count_full_value=%d '
                      'count_short_value=%d '
                      'n=%d num_all_other=%d '
                      'm_valid=%d' % (
                          thecount,
                          count_full_value,
                          count_short_value,
                          n,
                          num_all_other,
                          m_valid,
                      ))
                sys.stdout.flush()
            self.valid_pred_vs_alphanew = \
                np.fromiter(cast(valid_pred_vs_alpha,
                                 POINTER(self.myctype)), dtype=self.dtype,
                            count=thecount)
            self.valid_pred_vs_alphanew = \
                np.reshape(self.valid_pred_vs_alphanew, (self.n_alphas,
                                                         m_valid))
            self.valid_pred_vs_alphapure = \
                self.valid_pred_vs_alphanew[:, 0:m_valid]

        return self

    # pylint: disable=unused-argument
    def predict_ptr(self,
                    valid_xptr=None,
                    valid_yptr=None,
                    free_input_data=0,
                    order=None):
        """Predict on a fitted GLM with with pointers to data on the GPU

        :param ndarray valid_xptr : Pointer to validation features

        :param ndarray valid_ yptr : Pointer to validation response

        :param int store_full_path : Store full regularization path
            from glm model

        :param int free_input_data : Indicate if input data should be freed
            at the end of fit(). Default is 1.

        :param int verbose : Print verbose information to the console
            if set to > 0. Default is 0.

        :param order: Order of data.  Default is None, and internally determined
        whether row 'r' or column 'c' major order.
        """

        #assume self.ord already set by fit_ptr() at least
        #override self if chose to pass this option
        oldstorefullpath = self.store_full_path
        if self.store_full_path == 1:  # then need to run twice
            self.store_full_path = 1
            self._fitorpredict_ptr(
                self.source_dev,
                self.m_train,
                self.n,
                self.m_valid,
                self.double_precision,
                self.ord,
                self.a,
                self.b,
                valid_xptr,
                valid_yptr,
                self.e,
                do_predict=1,
                free_input_data=free_input_data,
            )
        self.store_full_path = 0
        self._fitorpredict_ptr(
            self.source_dev,
            self.m_train,
            self.n,
            self.m_valid,
            self.double_precision,
            self.ord,
            self.a,
            self.b,
            valid_xptr,
            valid_yptr,
            self.e,
            do_predict=1,
        )
        #restore global variable
        self.store_full_path = oldstorefullpath

        return self.valid_pred_vs_alphapure  # something like valid_y

    # pylint: disable=unused-argument
    def fit_predict(self,
                    train_x,
                    train_y,
                    valid_x=None,
                    valid_y=None,
                    sample_weight=None,
                    free_input_data=1,
                    order=None):
        """Train a model using GLM and predict on validation set

        :param ndarray train_x : Training features array

        :param ndarray train_ y : Training response array

        :param ndarray valid_x : Validation features

        :param ndarray valid_ y : Validation response

        :param ndarray weight : Observation weights

        :param int free_input_data : Indicate if input data should be freed at
            the end of fit(). Default is 1.

        :param order: Order of data.  Default is None, and internally determined
            whether row 'r' or column 'c' major order.
        """

        #let fit() check and convert(to numpy)
        #train_x, train_y, valid_x, valid_y, weight
        self.fit(
            train_x,
            train_y,
            valid_x,
            valid_y,
            sample_weight,
            free_input_data=0,
        )
        if valid_x is None:
            self.prediction = self.predict(
                valid_x=train_x,
                valid_y=train_y,
                sample_weight=sample_weight,
                free_input_data=free_input_data)
        else:
            self.prediction = self.predict(
                valid_x=valid_x,
                valid_y=valid_y,
                sample_weight=sample_weight,
                free_input_data=free_input_data)
        return self.prediction  # something like valid_y

    # pylint: disable=unused-argument
    def fit_predict_ptr(self,
                        m_train,
                        n,
                        m_valid,
                        double_precision,
                        order,
                        a,
                        b,
                        c,
                        d,
                        e,
                        free_input_data=0,
                        source_dev=0):
        """Train a GLM with pointers to data on the GPU and predict
        on validation set that also has a pointer on the GPU

        :param m_train Number of rows in the training set

        :param n Number of columns in the training set

        :param m_valid Number of rows in the validation set

        :param double_precision float32 (0) or double precision (1) of fit.
            Default None.

        :param order: Order of data.  Default is None, and internally determined
        whether row 'r' or column 'c' major order.

        :param a Pointer to training features array

        :param b Pointer to training response array

        :param c Pointer to validation features

        :param d Pointer to validation response

        :param e Pointer to weight column

        :param int free_input_data : Indicate if input data should be freed
            at the end of fit(). Default is 1.

        :param source_dev GPU ID of device

        """

        do_predict = 0  # only fit at first

        self._fitorpredict_ptr(
            source_dev,
            m_train,
            n,
            m_valid,
            double_precision,
            self.ord,
            a,
            b,
            c,
            d,
            e,
            do_predict,
            free_input_data=0)
        if c is None or c is c_void_p(0):
            self.prediction = self.predict_ptr(
                valid_xptr=a, valid_yptr=b, free_input_data=free_input_data)
        else:
            self.prediction = self.predict_ptr(
                valid_xptr=c, valid_yptr=d, free_input_data=free_input_data)
        return self.prediction  # something like valid_y

    def fit_transform(self,
                      train_x,
                      train_y,
                      valid_x=None,
                      valid_y=None,
                      sample_weight=None,
                      free_input_data=1):
        """Train a model using GLM and predict on validation set

        :param ndarray train_x : Training features array

        :param ndarray train_ y : Training response array

        :param ndarray valid_x : Validation features

        :param ndarray valid_ y : Validation response

        :param ndarray weight : Observation weights

        :param int free_input_data : Indicate if input data should be freed at
            the end of fit(). Default is 1.
        """

        return self.fit_predict(self, train_x, train_y, valid_x, valid_y,
                                sample_weight, free_input_data)

    def transform(self):
        return self

    def summary(self):
        """
        Obtain model summary, which is error per alpha across train,
        cv, and validation

        Error is logloss for classification and
        RMSE (Root Mean Squared Error) for regression.
        """
        error_train = pd.DataFrame(self.error_best, index=self.alphas)
        if self.family == "logistic":
            print("Logloss per alpha value (-1.00 = missing)\n")
        else:
            print("RMSE per alpha value (-1.00 = missing)\n")
        headers = ["Alphas", "Train", "CV", "Valid"]
        print(
            tabulate(
                error_train, headers=headers, tablefmt="pipe", floatfmt=".2f"))

# ################## #Properties and setters of properties

    @property
    def total_n_gpus(self):
        return self._total_n_gpus

    @property
    def gpu_id(self):
        return self._gpu_id

    @gpu_id.setter
    def gpu_id(self, value):
        assert value >= 0, "GPU ID must be non-negative."
        self._gpu_id = value

    @property
    def family(self):
        return self._family_str

    @family.setter
    def family(self, value):
        #add check
        self.family = value

    @property
    def shared_a(self):
        return self._shared_a

    @shared_a.setter
    def shared_a(self, value):
        #add check
        self.__shared_a = value

    @property
    def standardize(self):
        return self._standardize

    @standardize.setter
    def standardize(self, value):

        #add check
        self._standardize = value

    @property
    def coef_(self):
        return self.x_vs_alphapure

    @property
    def X(self):
        return self.x_vs_alphapure

    @property
    def X_full(self):
        ''' Returns full solution if store_full_path=1
           X[which lambda][which alpha]
         '''
        return self.x_vs_alpha_lambdapure

    @property
    def X_best(self):
        return self.x_vs_alphapure

    @property
    def validPreds(self):
        return self.valid_pred_vs_alphapure

    @property
    def validPreds_best(self):
        return self.valid_pred_vs_alphapure

    @property
    def intercept_(self):
        return self.intercept2_

    @intercept_.setter
    def intercept_(self, value):
        self._intercept_ = value

    @property
    def intercept_best(self):
        return self.intercept2_

    @property
    def error(self):
        return self.error_vs_alpha

    @property
    def lambdas(self):
        return self._lambdas2

    @lambdas.setter
    def lambdas(self, value):
        self._lambdas = value

    @property
    def alphas(self):
        return self._alphas2

    @alphas.setter
    def alphas(self, value):
        self._alphas = value

    @property
    def tols(self):
        return self._tols2

    @tols.setter
    def tols(self, value):
        self._tols = value

    @property
    def validPreds_full(self):
        ''' Returns full predictions if store_full_path=1
           validPreds[which lambda][which alpha]
         '''
        return self.valid_pred_vs_alpha_lambdapure

    @property
    def intercept_full(self):
        ''' Returns full intercept if store_full_path=1
           intercept[which lambda][which alpha]
         '''
        return self.intercept_

    @property
    def error_full(self):
        return self.error_vs_alpha_lambda

    @property
    def lambdas_full(self):
        ''' Returns full lambda path if store_full_path=1
           lambda[which lambda][which alpha]
         '''
        return self._lambdas

    @property
    def alphas_full(self):
        ''' Returns full alpha if store_full_path=1
           alpha[which lambda][which alpha]
         '''
        return self._alphas

    @property
    def tols_full(self):
        ''' Returns full tols if store_full_path=1
           tols[which lambda][which alpha]
         '''
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

#     def score(self, X=None, y=None, sample_weight=None):
#         if X is not None and y is not None:
#             self.prediction = self.predict(
#                 valid_x=X, valid_y=y, sample_weight=sample_weight)
# #otherwise score makes no sense, need both X and y,
# #else just return existing error
# #TODO : Should return R ^ 2 and redo predict if X and y are passed
#         return self.error

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        #fetch the constructor or the original constructor before
        #deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            #No explicit constructor to introspect
            return []

#introspect the constructor arguments to find the model parameters
#to represent
        from ..utils.fixes import signature
        init_signature = signature(init)
        #Consider the constructor parameters excluding 'self'
        parameters = [
            p for p in init_signature.parameters.values()
            if p.name != 'self' and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("h2o4gpu GLM estimator should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention." %
                                   (cls, init_signature))
#Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        :param bool deep : If True, will return the parameters for this
            estimator and contained subobjects that are estimators.

        :returns dict params : Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            #We need deprecation warnings to always be on in order to
            #catch deprecated param values.
            #This is set in utils / __init__.py but it gets overwritten
            #when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if w and w[0].category == DeprecationWarning:
                    #if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)


#XXX : should we rather test if instance of estimator ?
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        if not params:
            #Simple optimization to gain speed(inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        from ..externals import six
        for key, value in six.iteritems(params):
            split = key.split('__', 1)
            if len(split) > 1:
                #nested objects case
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (name, self))
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                #simple objects case
                if key not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (key, self.__class__.__name__))
                setattr(self, key, value)
        return self


class ElasticNet(object):
    """H2O ElasticNet Solver

    Selects between h2o4gpu.solvers.elastic_net.ElasticNet_h2o4gpu
    and h2o4gpu.linear_model.coordinate_descent.ElasticNet_sklearn

    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the penalty terms. Defaults to 1.0.
        See the notes for the exact mathematical meaning of this
        parameter.``alpha = 0`` is equivalent to an ordinary least square,
        solved by the :class:`LinearRegressionSklearn` object. For numerical
        reasons, using ``alpha = 0`` with the ``LassoSklearn`` object is not advised.
        Given this, you should use the :class:`LinearRegressionSklearn` object.

    l1_ratio : float
        The ElasticNetSklearn mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.

    fit_intercept : bool
        Whether the intercept should be estimated or not. If ``False``, the
        data is assumed to be already centered.

    normalize : boolean, optional, default False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`h2o4gpu.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    precompute : True | False | array-like
        Whether to use a precomputed Gram matrix to speed up
        calculations. The Gram matrix can also be passed as argument.
        For sparse input this option is always ``True`` to preserve sparsity.

    max_iter : int, optional
        The maximum number of iterations

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.

    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    warm_start : bool, optional
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    positive : bool, optional
        When set to ``True``, forces the coefficients to be positive.

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator that selects a random
        feature to update.  If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`. Used when ``selection`` ==
        'random'.

    selection : str, default 'cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    n_gpus : int, (Default=-1)
        Number of gpu's to use in GLM solver.

    lambda_stop_early : float, (Default=True)
        Stop early when there is no more relative
        improvement on train or validation.

    glm_stop_early : bool, (Default=True)
        Stop early when there is no more relative
        improvement in the primary and dual residuals for ADMM.

    glm_stop_early_error_fraction : float, (Default=1.0)
        Relative tolerance for metric-based stopping criterion (stop if relative improvement is not at
        least this much).

    verbose : int, (Default=0)
        Print verbose information to the console if set to > 0.

    n_threads : int, (Default=None)
        Number of threads to use in the gpu.
        Each thread is an independent model builder.

    gpu_id : int, optional, (default=0)
        ID of the GPU on which the algorithm should run.

    lambda_min_ratio: float, (Default=1E-7).
        Minimum lambda ratio to maximum lambda, used
        in lambda search.

    n_lambdas : int, (Default=100)
        Number of lambdas to be used in a search.

    n_folds : int,  (Default=1)
        Number of cross validation folds.

    n_alphas : int, (Default=5)
        Number of alphas to be used in a search.

    tol_seek_factor : float, (Default=1E-1)
        Factor of tolerance to seek
        once below null model accuracy.  Default is 1E-1, so seeks tolerance
        of 1E-3 once below null model accuracy for tol=1E-2.

    family : string, (Default="elasticnet")
        "logistic" for classification with logistic regression.
        Defaults to "elasticnet" for regression.
        Must be "logistic" or "elasticnet".

    store_full_path: int, (Default=0)
        Whether to store full solution for all alphas
        and lambdas.  If 1, then during predict will compute best
        and full predictions.

    lambda_max : int, (Default=None)
        Maximum Lambda value to use.
        Default is None, and then internally compute standard maximum

    alpha_max : float, (Default=1.0)
        Maximum alpha.

    alpha_min : float, (Default=0.0)
        Minimum alpha.

    alphas: list, tuple, array, or numpy 1D array of alphas (Default=None)
        overrides n_alphas, alpha_min, and alpha_max.

    lambdas: list, tuple, array, or numpy 1D array of lambdas (Default=None)
        overrides n_lambdas, lambda_max, and lambda_min_ratio.

    double_precision: int, (Default=None)
        Internally set unless using _ptr methods. Value can either be
        0 (float32) or 1(float64)

    order : string, (Default=None)
        Order of data. Default is None, and internally
        determined (unless using _ptr methods) whether
        row 'r' or column 'c' major order.

    backend : string, (Default="auto")
        Which backend to use.
        Options are 'auto', 'sklearn', 'h2o4gpu'.
        Saves as attribute for actual backend used.

    """
    def __init__(
            self,
            alpha=1.0, #scikit
            l1_ratio=0.5, #scikit
            fit_intercept=True, #h2o4gpu and scikit
            normalize=False, #scikit
            precompute=False, #scikit
            max_iter=5000, #scikit
            copy_X=True, #scikit
            tol=1e-2, #h2o4gpu and scikit
            warm_start=False, #scikit
            positive=False, #scikit
            random_state=None, #scikit
            selection='cyclic', #scikit
            n_gpus=-1,  # h2o4gpu
            lambda_stop_early=True,  # h2o4gpu
            glm_stop_early=True,  # h2o4gpu
            glm_stop_early_error_fraction=1.0,  #h2o4gpu
            verbose=False, #h2o4gpu
            n_threads=None, #h2o4gpu
            gpu_id=0, #h2o4gpu
            lambda_min_ratio=1E-7, #h2o4gpu
            n_lambdas=100, #h2o4gpu
            n_folds=5, #h2o4gpu
            n_alphas=5, #h2o4gpu
            tol_seek_factor=1E-1, #h2o4gpu
            family='elasticnet', #h2o4gpu
            store_full_path=0, #h2o4gpu
            lambda_max=None, #h2o4gpu
            alpha_max=1.0, #h2o4gpu
            alpha_min=0.0, #h2o4gpu
            alphas=None, #h2o4gpu
            lambdas=None, #h2o4gpu
            double_precision=None, #h2o4gpu
            order=None, #h2o4gpu
            backend='auto'):  # h2o4gpu

        import os
        _backend = os.environ.get('H2O4GPU_BACKEND', None)
        if _backend is not None:
            backend = _backend

        # Fall back to Sklearn
        # Can remove if fully implement sklearn functionality
        # If parameter not listed, then ignored because not important
        self.do_sklearn = False
        if backend == 'auto':

            params_string = ['alpha', 'l1_ratio', 'normalize', 'precompute',
                             'max_iter', 'copy_X',
                             'warm_start', 'positive',
                             'random_state', 'selection']
            params = [alpha, l1_ratio, normalize, precompute,
                      max_iter, copy_X,
                      warm_start, positive,
                      random_state, selection]
            params_default = [1.0, 0.5, False, False, 5000, True,
                              False, False, None, 'cyclic']

            i = 0
            for param in params:
                if param != params_default[i]:
                    self.do_sklearn = True
                    if verbose:
                        print("WARNING:"
                              " The sklearn parameter " + params_string[i] +
                              " has been changed from default to " +
                              str(param) + ". Will use Sklearn.")
                    self.do_sklearn = True
                i = i + 1
        elif backend == 'sklearn':
            self.do_sklearn = True
        elif backend == 'h2o4gpu':
            self.do_sklearn = False
        if self.do_sklearn:
            self.backend = 'sklearn'
        else:
            self.backend = 'h2o4gpu'

        self.model_sklearn = sk.ElasticNetSklearn(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            normalize=normalize,
            precompute=precompute,
            max_iter=max_iter,
            copy_X=copy_X,
            tol=tol,
            warm_start=warm_start,
            positive=positive,
            random_state=random_state,
            selection=selection)

        self.model_h2o4gpu = ElasticNetH2O(
            gpu_id=gpu_id,
            tol_seek_factor=tol_seek_factor,
            family=family,
            n_threads=n_threads,
            n_gpus=n_gpus,
            double_precision=double_precision,
            fit_intercept=fit_intercept,
            lambda_min_ratio=lambda_min_ratio,
            n_lambdas=n_lambdas,
            n_folds=n_folds,
            n_alphas=n_alphas,
            tol=tol,
            lambda_stop_early=lambda_stop_early,
            glm_stop_early=glm_stop_early,
            glm_stop_early_error_fraction=glm_stop_early_error_fraction,
            max_iter=max_iter,
            verbose=verbose,
            store_full_path=store_full_path,
            lambda_max=lambda_max,
            alpha_max=alpha_max,
            alpha_min=alpha_min,
            alphas=alphas,
            lambdas=lambdas,
            order=order)

        if self.do_sklearn:
            if verbose:
                print("Running sklearn Lasso Regression")
            self.model = self.model_sklearn
        else:
            if verbose:
                print("Running h2o4gpu Lasso Regression")
            self.model = self.model_h2o4gpu

        self.verbose = verbose

    def fit(self, X, y=None, check_input=True):
        if self.do_sklearn:
            res = self.model.fit(X, y, check_input)
            self.set_attributes()
            return res
        res = self.model.fit(X, y)
        self.set_attributes()
        return res

    def get_params(self):
        return self.model.get_params()

    def predict(self, X):
        res = self.model.predict(X)
        self.set_attributes()
        return res

    def predict_proba(self, X):
        res = self.model.predict_proba(X)
        self.set_attributes()
        return res

    def score(self, X, y, sample_weight=None):
        # TODO: add for h2o4gpu
        if self.verbose:
            print("WARNING: score() is using sklearn")
        if not self.do_sklearn:
            self.model_sklearn.fit(X, y)  #Need to re-fit
        res = self.model_sklearn.score(X, y, sample_weight)
        return res

    def set_params(self, **params):
        return self.model.set_params(**params)

    def set_attributes(self):
        """
        Set attributes and don't fail if not yet present
        """
        s = _setter(oself=self, e1=NameError, e2=AttributeError)

        self.coef_ = None
        s('oself.coef_ = oself.model.coef_')
        self.sparse_coef_ = None
        s('oself.sparse_coef_ = oself.model.sparse_coef_')
        self.intercept_ = None
        s('oself.intercept_ = oself.model.intercept_')
        self.n_iter_ = None
        s('oself.n_iter_ = oself.model.n_iter_')

        self.time_prepare = None
        s('oself.time_prepare = oself.model.time_prepare')
        self.time_upload_data = None
        s('oself.time_upload_data = oself.model.time_upload_data')
        self.time_fitonly = None
        s('oself.time_fitonly = oself.model.time_fitonly')
