# -*- encoding: utf-8 -*-
"""
:copyright: 2017-2018 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import sys
import time
import numpy as np

# Data utils

class WrappedPointer(object):
    '''Wraps a native pointer and release underlying resources
    when notified by GC

    Arguments:
        p {[type]} -- pointer
        double_precision {bool} -- use double precision
        lib{object} -- python wrapper over lib.so
    '''

    def __init__(self, p, double_precision, lib):
        self.p = p
        self.double_precision = double_precision
        self.lib = lib
    def __del__(self):
        if self.double_precision:
            self.lib.modelfree1_double(self.p)
        else:
            self.lib.modelfree1_float(self.p)

def _get_order(data, fortran, order):
    """ Return the Unicode code point representing the
    order of this data set. """
    if data is not None:
        if order is None:
            order = 'c' if fortran else 'r'
        elif order in ['c', 'r']:
            order = order
        elif order in ['c', 'r']:
            order = order
        else:
            ValueError("Bad order")
    return order


def _to_np(data, ismatrix=False, dtype=None, order=None):
    """Convert the input to a numpy array.

    :param data: array_like
    :return: ndarray
    """

    # handle pandas input
    # TODO: Store pandas names at least and attach back to X/coef for output
    import pandas as pd
    if isinstance(data, pd.DataFrame):
        outdata = data.values
    elif isinstance(data, np.ndarray):
        outdata = data
    else:
        outdata = np.asarray(data)

    # deal with degenerate matrices
    if ismatrix and len(outdata.shape) == 1:
        nrows = outdata.shape[0]
        ncols = 1
        outdata = outdata.reshape((nrows, ncols))

    # convert to correct precision if necessary
    if order is not None:
        if order == 'r':
            nporder = 'C'
        elif order == 'c':
            nporder = 'F'
        else:
            nporder = 'C'
            ValueError("No such order")
    else:
        # in case both (i.e. 1D array), then default to C order
        if outdata.flags.c_contiguous:
            nporder = 'C'
        else:
            nporder = 'F'
    if dtype is None:
        dtype = outdata.dtype

    # force precision as 32-bit if not required types
    if dtype != np.float32 and dtype != np.float64:
        dtype = np.float32

    outdata = outdata.astype(dtype, copy=False, order=nporder)

    selford = _get_order(
        outdata, fortran=not outdata.flags.c_contiguous, order=order)

    return outdata, selford, dtype


def munge(data_as_np, fit_intercept=False):
    """Munge Data
    """
    # If True, then append intercept term to
    # train_x array and valid_x array(if available)
    #
    # Not this is really munging as adds to columns
    # and changes expected size of outputted solution
    if (fit_intercept or fit_intercept == 1) and len(data_as_np.shape) == 2:
        data_as_np = np.hstack([
            data_as_np,
            np.ones((data_as_np.shape[0], 1), dtype=data_as_np.dtype)
        ])
    return data_as_np


def _get_data(data, ismatrix=False, fit_intercept=False, order=None,
              dtype=None):
    """Transforms data to numpy and gather basic info about it.

    :param data: array_like
    :return: data as ndarray, rows, cols, continuity
    """
    # default is no data
    data_as_np = None  # specific to this data
    m = -1  # specific to this data
    n = -1  # specific to this data
    fortran = None  # specific to this data
    # dtype and order not specific to this data, can be just input

    if data is not None:
        data_as_np, order, dtype = _to_np(
            data, ismatrix=ismatrix, dtype=dtype, order=order)
        data_as_np = munge(data_as_np, fit_intercept=fit_intercept)
        fortran = not data_as_np.flags.c_contiguous
        shape_x = np.shape(data_as_np)
        m = shape_x[0]
        if len(shape_x) > 1:
            n = shape_x[1]
        else:
            n = 1

    return data_as_np, m, n, fortran, order, dtype


def _check_data_content(do_check, name, data):
    """Makes sure the data contains no infinite or NaN values

    :param do_check: int
        1 perform checks
        != 1 don't perform checks.
    :param name: str
        Name of the object for logging.
    :param data: array_like
        Data to be checked
    :return:
    """
    if do_check == 1:
        assert np.isfinite(data).all(), "%s contains Inf" % name
        assert not np.isnan(data).any(), "%s contains NA" % name


def _data_info(data, verbose=0):
    """Get info about passed data.

    :param data: numpy-only array
    :param verbose: int, optional, default : 0
        Logging level
    :return:
        Data precision (0 or 1), rows, cols
    """
    double_precision = -1
    m = 0
    n = -1

    if data is not None:
        try:
            if data.dtype == np.float64:
                if verbose > 0:
                    print('Detected np.float64 data')
                    sys.stdout.flush()
                double_precision = 1
            if data.dtype == np.float32:
                if verbose > 0:
                    print('Detected np.float32 data')
                    sys.stdout.flush()
                double_precision = 0
        except AttributeError:
            double_precision = -1

        data_shape = np.shape(data)
        if len(data_shape) == 1:
            m = data_shape[0]
        elif len(data_shape) == 2:
            m = data_shape[0]
            n = data_shape[1]

    return double_precision, m, n

def _check_equal(iterator):
    """Check if all the values in an iterator are equal.

    :param iterator: iterator
    :return: bool
    """

    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)


def prepare_and_upload_data(self,
                            train_x=None,
                            train_y=None,
                            valid_x=None,
                            valid_y=None,
                            sample_weight=None,
                            source_dev=0):
    """ Prepare data and then upload data
    """
    time_prepare0 = time.time()
    train_x_np, m_train, n1, fortran1, self.ord, self.dtype = _get_data(
        train_x,
        ismatrix=True,
        fit_intercept=self.fit_intercept,
        order=self.ord,
        dtype=self.dtype)
    train_y_np, m_y, _, fortran2, self.ord, self.dtype = _get_data(
        train_y, order=self.ord, dtype=self.dtype)
    valid_x_np, m_valid, n2, fortran3, self.ord, self.dtype = _get_data(
        valid_x,
        ismatrix=True,
        fit_intercept=self.fit_intercept,
        order=self.ord,
        dtype=self.dtype)
    valid_y_np, m_valid_y, _, fortran4, self.ord, self.dtype = \
        _get_data(valid_y, order=self.ord, dtype=self.dtype)
    weight_np, _, _, fortran5, self.ord, self.dtype = _get_data(
        sample_weight, order=self.ord, dtype=self.dtype)

    # check that inputs all have same 'c' or 'r' order
    fortran_list = [fortran1, fortran2, fortran3, fortran4, fortran5]
    _check_equal(fortran_list)

    # now can do checks

    # ############## #
    # check do_predict input

    if m_train >= 1 and m_y >= 1 and m_train != m_y:
        print('training X and Y must have same number of rows, '
              'but m_train=%d m_y=%d\n' % (m_train, m_y))

    # ################

    if n1 >= 0 and n2 >= 0 and n1 != n2:
        raise ValueError(
            'train_x and valid_x must have same number of columns, '
            'but n=%d n2=%d\n' % (n1, n2))

    # ################ #

    if m_valid >= 0 and m_valid_y >= 0 and m_valid != m_valid_y:
        raise ValueError('valid_x and valid_y must have same number of rows, '
                         'but m_valid=%d m_valid_y=%d\n' % (m_valid, m_valid_y))
        # otherwise m_valid is used, and m_valid_y can be there
    # or not(sets whether do error or not)
    self.time_prepare = time.time() - time_prepare0

    time_upload_data0 = time.time()
    (a, b, c, d, e) = upload_data(self, train_x_np, train_y_np, valid_x_np,
                                  valid_y_np, weight_np, source_dev)

    self.time_upload_data = time.time() - time_upload_data0

    self.a = a
    self.b = b
    self.c = c
    self.d = d
    self.e = e
    return a, b, c, d, e


def upload_data(self,
                train_x,
                train_y,
                valid_x=None,
                valid_y=None,
                sample_weight=None,
                source_dev=0):
    """Upload the data through the backend library"""

    self.double_precision1, m_train, n1 = _data_info(train_x, self.verbose)
    self.m_train = m_train
    self.double_precision3, _, _ = _data_info(train_y, self.verbose)
    self.double_precision2, m_valid, n2 = _data_info(valid_x, self.verbose)
    self.m_valid = m_valid
    self.double_precision4, _, _ = _data_info(valid_y, self.verbose)
    self.double_precision5, _, _ = _data_info(sample_weight, self.verbose)

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

    # ############## #

    if self.double_precision == 1:
        self.dtype = np.float64

        if self.verbose > 0:
            print('Detected np.float64')
            sys.stdout.flush()
    else:
        self.dtype = np.float32

        if self.verbose > 0:
            print('Detected np.float32')
            sys.stdout.flush()

    a = None
    b = None
    c = None
    d = None
    e = None

    A = train_x
    B = train_y
    C = valid_x
    D = valid_y
    E = sample_weight

    if self.double_precision == 1:
        c_upload_data = self.lib.make_ptr_double
    elif self.double_precision == 0:
        c_upload_data = self.lib.make_ptr_float
    else:
        print('Unknown numpy type detected')
        print(train_x.dtype)
        sys.stdout.flush()
        return a, b, c, d, e

    status, a, b, c, d, e = c_upload_data(
        self._shared_a,  # pylint: disable=W0212
        self.source_me,
        source_dev,
        m_train,
        n,
        m_valid,
        self.ord,
        A,
        B,
        C,
        D,
        E,
        a,
        b,
        c,
        d,
        e
    )

    assert status == 0, 'Failure uploading the data'

    # self.a = a
    # self.b = b
    # self.c = c
    # self.d = d
    # self.e = e
    return WrappedPointer(a, self.double_precision == 1, self.lib),\
           WrappedPointer(b, self.double_precision == 1, self.lib),\
           WrappedPointer(c, self.double_precision == 1, self.lib),\
           WrappedPointer(d, self.double_precision == 1, self.lib),\
           WrappedPointer(e, self.double_precision == 1, self.lib)


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
    import warnings
    warnings.warn("finish will be removed in a next version, please use 'del'",
                  DeprecationWarning, stacklevel=2)
    free_sols(self)
    free_preds(self)

def __del__(self):
    free_sols(self)
    free_preds(self)

class _setter:
    """Setter
    """

    def __init__(self, oself, e1, e2):
        self._e1 = e1
        self._e2 = e2
        self.oself = oself

    def __call__(self, expression):
        try:
            # pylint: disable=unused-variable
            oself = self.oself
            # pylint: disable=exec-used
            exec(expression)
        except self._e1:
            pass
        except self._e2:
            pass
