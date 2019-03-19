# pylint: skip-file
from .util import import_data
from .util import metrics
from . import h2o4gpu_exceptions
from .typecheck import compatibility
from .typecheck import typechecks
from .solvers.truncated_svd import TruncatedSVDH2O
from .solvers.truncated_svd import TruncatedSVD
from .solvers.pca import PCAH2O
from .solvers.pca import PCA
from .solvers.kmeans import KMeansH2O
from .solvers.kmeans import KMeans
from .solvers.xgboost import GradientBoostingRegressor
from .solvers.xgboost import GradientBoostingClassifier
from .solvers.xgboost import RandomForestClassifier
from .solvers.xgboost import RandomForestRegressor
from .solvers.ridge import Ridge
from .solvers.lasso import Lasso
from .solvers.linear_regression import LinearRegression
from .solvers.logistic import LogisticRegression
from .solvers.elastic_net import ElasticNetH2O
from .solvers.elastic_net import ElasticNet
from .solvers.pogs import Pogs
from .types import FunctionVector

# Skip pylint b / c this is automatically concatenated at compile time
# with other init files

DAAL_SUPPORTED = True

try:
    __import__('daal')
except ImportError:
    DAAL_SUPPORTED = False

if DAAL_SUPPORTED:
    from .solvers.daal_solver.regression import Method as LinearMethod
