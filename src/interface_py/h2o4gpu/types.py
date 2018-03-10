"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
from ctypes import POINTER, c_int, c_int32, c_uint, c_void_p, c_float, \
    c_double, Structure
from numpy import zeros, ones, inf


class H2OConstants:
    COL_MAJ = c_int(0)
    ROW_MAJ = c_int(1)


class H2OFunctions:
    """
    Constants representing functions used internally in C/C++
    """
    ABS = c_int(0)
    EXP = c_int(1)
    HUBER = c_int(2)
    IDENTITY = c_int(3)
    INDBOX01 = c_int(4)
    INDEQ0 = c_int(5)
    INDGE0 = c_int(6)
    INDLE0 = c_int(7)
    LOGISTIC = c_int(8)
    MAXNEG0 = c_int(9)
    MAXPOS0 = c_int(10)
    NEGENTR = c_int(11)
    NEGLOG = c_int(12)
    RECIPR = c_int(13)
    SQUARE = c_int(14)
    ZERO = c_int(15)


class H2OStatus:
    """
    Constants representing statuses returned from C/C++
    """
    SUCCESS = 'H2O4GPU_SUCCESS'
    INFEASIBLE = 'H2O4GPU_INFEASIBLE'
    UNBOUNDED = 'H2O4GPU_UNBOUNDED'
    MAX_ITER = 'H2O4GPU_MAX_ITER'
    NAN_FOUND = 'H2O4GPU_NAN_FOUND'
    ERROR = 'H2O4GPU_ERROR'


class H2OSolverDefault:
    """
    Constants representing defaults used in our solvers
    """
    RHO = 1.  # rho = 1.0
    ABS_TOL = 1e-4  # abs_tol = 1e-2
    REL_TOL = 1e-4  # rel_tol = 1e-4
    MAX_ITERS = 2500  # max_iters = 2500
    VERBOSE = 2  # verbose = 2
    ADAPTIVE_RHO = 1  # adaptive_rho = True
    EQUIL = 1  # equil = True
    GAP_STOP = 1  # gap_stop = True
    WARM_START = 0  # warm_start = False
    N_DEV = 1  # number of cuda devices =1
    W_DEV = 0  # which cuda devices (0)


#pointers to C types
c_int_p = POINTER(c_int)
c_int32_p = POINTER(c_int32)
c_float_p = POINTER(c_float)
c_void_pp = POINTER(c_void_p)
c_double_p = POINTER(c_double)


#H2O4GPU types
class SettingsS(Structure):
    _fields_ = [('rho', c_float), ('abs_tol', c_float), ('rel_tol', c_float),
                ('max_iters', c_uint), ('verbose', c_uint),
                ('adaptive_rho', c_int), ('equil', c_int), ('gap_stop', c_int),
                ('warm_start', c_int), ('nDev', c_int), ('wDev', c_int)]


class SettingsD(Structure):
    _fields_ = [('rho', c_double), ('abs_tol', c_double), ('rel_tol', c_double),
                ('max_iters', c_uint), ('verbose', c_uint),
                ('adaptive_rho', c_int), ('equil', c_int), ('gap_stop', c_int),
                ('warm_start', c_int), ('nDev', c_int), ('wDev', c_int)]


class InfoS(Structure):
    _fields_ = [('iter', c_uint), ('status', c_int), ('obj', c_float),
                ('rho', c_float), ('solvetime', c_float)]


class InfoD(Structure):
    _fields_ = [('iter', c_uint), ('status', c_int), ('obj', c_double),
                ('rho', c_double), ('solvetime', c_float)]


class Solution(object):

    def __init__(self, double_precision, m, n):
        T = c_double if double_precision else c_float
        self.double_precision = double_precision
        self.x = zeros(n, dtype=T)
        self.y = zeros(m, dtype=T)
        self.mu = zeros(n, dtype=T)
        self.nu = zeros(m, dtype=T)


class SolutionS(Structure):
    _fields_ = [('x', c_float_p), ('y', c_float_p), ('mu', c_float_p),
                ('nu', c_float_p)]


class SolutionD(Structure):
    _fields_ = [('x', c_double_p), ('y', c_double_p), ('mu', c_double_p),
                ('nu', c_double_p)]


#pointers to H2O4GPU types
settings_s_p = POINTER(SettingsS)
settings_d_p = POINTER(SettingsD)
info_s_p = POINTER(InfoS)
info_d_p = POINTER(InfoD)
solution_s_p = POINTER(SolutionS)
solution_d_p = POINTER(SolutionD)


def cptr(np_arr, dtype=c_float):
    return np_arr.ctypes.data_as(POINTER(dtype))


def change_settings(settings, **kwargs):
    """ Utility setting values from kwargs
    :param settings: settings object, should contain attributes
        which are about to be set
    :param kwargs: key-value pairs representing the settings
    :return:
    """
    #all settings(except warm_start) are persistent and change only if called
    if 'rho' in kwargs: settings.rho = kwargs['rho']
    if 'abs_tol' in kwargs: settings.abs_tol = kwargs['abs_tol']
    if 'rel_tol' in kwargs: settings.rel_tol = kwargs['rel_tol']
    if 'max_iters' in kwargs: settings.max_iters = kwargs['max_iters']
    if 'verbose' in kwargs: settings.verbose = kwargs['verbose']
    if 'adaptive_rho' in kwargs: settings.adaptive_rho = kwargs['adaptive_rho']
    if 'equil' in kwargs: settings.equil = kwargs['equil']
    if 'gap_stop' in kwargs: settings.gap_stop = kwargs['gap_stop']

    #warm_start must be specified each time it is desired
    if 'warm_start' in kwargs:
        settings.warm_start = kwargs['warm_start']
    else:
        settings.warm_start = 0
    if 'nDev' in kwargs: settings.nDev = kwargs['nDev']
    if 'wDev' in kwargs: settings.wDev = kwargs['wDev']


def make_settings(double_precision=False, **kwargs):
    """Creates a SettingsS objects from key-values

    :param double_precision: boolean, optional, default : False
    :param kwargs: **kwargs
    :return: SettingsS object
    """
    rho = kwargs['rho'] if 'rho' in list(
        kwargs.keys()) else H2OSolverDefault.RHO
    relt = kwargs['abs_tol'] if 'abs_tol' in list(
        kwargs.keys()) else H2OSolverDefault.ABS_TOL
    abst = kwargs['rel_tol'] if 'rel_tol' in list(
        kwargs.keys()) else H2OSolverDefault.REL_TOL
    maxit = kwargs['max_iters'] if 'max_iters' in list(
        kwargs.keys()) else H2OSolverDefault.MAX_ITERS
    verb = kwargs['verbose'] if 'verbose' in list(
        kwargs.keys()) else H2OSolverDefault.VERBOSE
    adap = kwargs['adaptive_rho'] if 'adaptive_rho' in list(
        kwargs.keys()) else H2OSolverDefault.ADAPTIVE_RHO
    equil = kwargs['equil'] if 'equil' in list(
        kwargs.keys()) else H2OSolverDefault.EQUIL
    gaps = kwargs['gap_stop'] if 'gap_stop' in list(
        kwargs.keys()) else H2OSolverDefault.GAP_STOP
    warm = kwargs['warm_start'] if 'warm_start' in list(
        kwargs.keys()) else H2OSolverDefault.WARM_START
    ndev = kwargs['nDev'] if 'nDev' in list(
        kwargs.keys()) else H2OSolverDefault.N_DEV
    wdev = kwargs['wDev'] if 'wDev' in list(
        kwargs.keys()) else H2OSolverDefault.W_DEV
    if double_precision:
        return SettingsD(rho, relt, abst, maxit, verb, adap, equil, gaps, warm,
                         ndev, wdev)
    return SettingsS(rho, relt, abst, maxit, verb, adap, equil, gaps, warm,
                     ndev, wdev)


def change_solution(py_solution, **kwargs):
    try:
        if 'x_init' in kwargs: py_solution.x[:] = kwargs['x_init'][:]
        if 'nu_init' in kwargs: py_solution.nu[:] = kwargs['nu_init'][:]
    except:
        raise RuntimeError("Failed to change solution.")


def make_solution(py_solution):
    if py_solution.double_precision:
        return SolutionD(
            cptr(py_solution.x, c_double), cptr(py_solution.y, c_double),
            cptr(py_solution.mu, c_double), cptr(py_solution.nu, c_double))
    return SolutionS(
        cptr(py_solution.x, c_float), cptr(py_solution.y, c_float),
        cptr(py_solution.mu, c_float), cptr(py_solution.nu, c_float))


def make_info(double_precision):
    if double_precision:
        return InfoD(0, 0, inf, 0, 0)
    return InfoS(0, 0, inf, 0, 0)


class FunctionVector(object):
    """Class representing a function"""

    def __init__(self, length, double_precision=False):
        T = c_double if double_precision else c_float
        self.a = ones(length, T)
        self.b = zeros(length, T)
        self.c = ones(length, T)
        self.d = zeros(length, T)
        self.e = zeros(length, T)
        self.h = zeros(length, c_int)
        self.double_precision = double_precision

    def length(self):
        return len(self.a)

    def copy_from(self, f):
        self.a[:] = f.a[:]
        self.b[:] = f.b[:]
        self.c[:] = f.c[:]
        self.d[:] = f.d[:]
        self.e[:] = f.e[:]
        self.h[:] = f.h[:]

    def copy_to(self, f):
        f.a[:] = self.a[:]
        f.b[:] = self.b[:]
        f.c[:] = self.c[:]
        f.d[:] = self.d[:]
        f.e[:] = self.e[:]
        f.h[:] = self.h[:]

    def to_double(self):
        if self.double_precision:
            return self
        f = FunctionVector(self.length(), double_precision=True)
        self.copy_to(f)
        return f

    def to_float(self):
        if self.double_precision:
            f = FunctionVector(self.length())
            self.copy_to(f)
            return f
        return self
