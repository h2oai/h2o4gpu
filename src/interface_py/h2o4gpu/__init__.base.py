# pylint: skip-file
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
#Skip pylint b / c this is automatically concatenated at compile time
#with other init files
# TODO: grab this from BUILD_INFO.txt or __about__.py
__version__ = "0.2.0"

DAAL_SUPPORTED=True

try:
    __import__('daal')
except ImportError:
    DAAL_SUPPORTED=False

if DAAL_SUPPORTED:
    from .solvers.daal_solver.regression import Method as LinearMethod
from .types import FunctionVector
from .solvers.pogs import Pogs
from .solvers.elastic_net import ElasticNet
from .solvers.elastic_net import ElasticNetH2O
from .solvers.logistic import LogisticRegression
from .solvers.linear_regression import LinearRegression
from .solvers.lasso import Lasso
from .solvers.ridge import Ridge
from .solvers.xgboost import RandomForestRegressor
from .solvers.xgboost import RandomForestClassifier
from .solvers.xgboost import GradientBoostingClassifier
from .solvers.xgboost import GradientBoostingRegressor
from .solvers.kmeans import KMeans
from .solvers.kmeans import KMeansH2O
from .solvers.pca import PCA
from .solvers.pca import PCAH2O
from .solvers.truncated_svd import TruncatedSVD
from .solvers.truncated_svd import TruncatedSVDH2O
from .typecheck import typechecks
from .typecheck import compatibility
from . import h2o4gpu_exceptions
from .util import metrics
from .util import import_data
