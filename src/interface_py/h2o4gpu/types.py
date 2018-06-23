"""
:copyright: 2017-2018 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
from numpy import zeros, ones, inf
import h2o4gpu.libs.lib_utils as lib_utils

lib = None

def lazyLib():
    global lib
    if lib is None:
        from .util.gpu import device_count
        n_gpus, devices = device_count(n_gpus=-1)
        lib = lib_utils.get_lib(n_gpus, devices)
    return lib

class H2OSolverDefault(object):
    """
    Constants representing defaults used in our solvers
    """

    def __init__(self):
        pass

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

#H2O4GPU types
class Solution(object):

    def __init__(self, double_precision, m, n):
        T = np.float64 if double_precision else np.float32
        self.double_precision = double_precision
        self.x = zeros(n, dtype=T)
        self.y = zeros(m, dtype=T)
        self.mu = zeros(n, dtype=T)
        self.nu = zeros(m, dtype=T)

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
    settings = lazyLib().H2O4GPUSettingsD if double_precision \
        else lazyLib().H2O4GPUSettingsS
    settings.rho = kwargs['rho'] if 'rho' in list(
        kwargs.keys()) else H2OSolverDefault.RHO
    settings.relt = kwargs['abs_tol'] if 'abs_tol' in list(
        kwargs.keys()) else H2OSolverDefault.ABS_TOL
    settings.abst = kwargs['rel_tol'] if 'rel_tol' in list(
        kwargs.keys()) else H2OSolverDefault.REL_TOL
    settings.maxit = kwargs['max_iters'] if 'max_iters' in list(
        kwargs.keys()) else H2OSolverDefault.MAX_ITERS
    settings.verb = kwargs['verbose'] if 'verbose' in list(
        kwargs.keys()) else H2OSolverDefault.VERBOSE
    settings.adap = kwargs['adaptive_rho'] if 'adaptive_rho' in list(
        kwargs.keys()) else H2OSolverDefault.ADAPTIVE_RHO
    settings.equil = kwargs['equil'] if 'equil' in list(
        kwargs.keys()) else H2OSolverDefault.EQUIL
    settings.gaps = kwargs['gap_stop'] if 'gap_stop' in list(
        kwargs.keys()) else H2OSolverDefault.GAP_STOP
    settings.warm = kwargs['warm_start'] if 'warm_start' in list(
        kwargs.keys()) else H2OSolverDefault.WARM_START
    settings.ndev = kwargs['nDev'] if 'nDev' in list(
        kwargs.keys()) else H2OSolverDefault.N_DEV
    settings.wdev = kwargs['wDev'] if 'wDev' in list(
        kwargs.keys()) else H2OSolverDefault.W_DEV
    return settings

def change_solution(py_solution, **kwargs):
    try:
        if 'x_init' in kwargs: py_solution.x[:] = kwargs['x_init'][:]
        if 'nu_init' in kwargs: py_solution.nu[:] = kwargs['nu_init'][:]
    except:
        raise RuntimeError("Failed to change solution.")


def make_solution(py_solution):
    solution = lazyLib().H2O4GPUSolutionD() if py_solution.double_precision \
        else lazyLib().H2O4GPUSolutionS()
    solution.x = py_solution.x
    solution.y = py_solution.y
    solution.mu = py_solution.mu
    solution.nu = py_solution.nu
    return solution

def make_info(double_precision):
    info = lazyLib().H2O4GPUInfoD if double_precision \
        else lazyLib().H2O4GPUInfoS
    info.iter = 0
    info.status = 0
    info.obj = inf
    info.rho = 0
    info.solvetime = 0
    return info

class FunctionVector(object):
    """Class representing a function"""

    def __init__(self, length, double_precision=False):
        T = np.float64 if double_precision else np.float32
        self.a = ones(length, T)
        self.b = zeros(length, T)
        self.c = ones(length, T)
        self.d = zeros(length, T)
        self.e = zeros(length, T)
        self.h = zeros(length, np.int32)
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
