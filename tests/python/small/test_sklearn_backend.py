"""scikit-learn backend of the drop-in wrappers (CPU only, no GPU needed).

The wrappers keep their legacy scikit-learn era signatures but build the
overlaid scikit-learn 1.5.2 estimators, which dropped several of those
parameters and renamed some loss/criterion values. These tests pin the
translation layer so the CPU fallback keeps constructing and fitting.
"""
import numpy as np
import pytest

from h2o4gpu.solvers import ElasticNet
from h2o4gpu.solvers import GradientBoostingClassifier
from h2o4gpu.solvers import GradientBoostingRegressor
from h2o4gpu.solvers import KMeans
from h2o4gpu.solvers import Lasso
from h2o4gpu.solvers import LinearRegression
from h2o4gpu.solvers import LogisticRegression
from h2o4gpu.solvers import RandomForestClassifier
from h2o4gpu.solvers import RandomForestRegressor
from h2o4gpu.solvers import Ridge

_rng = np.random.RandomState(0)
X = _rng.rand(200, 6).astype(np.float32)
Y_REG = X @ np.arange(1, 7, dtype=np.float32)
Y_CLS = (Y_REG > np.median(Y_REG)).astype(int)

REGRESSORS = [ElasticNet, Lasso, Ridge, LinearRegression,
              RandomForestRegressor, GradientBoostingRegressor]
CLASSIFIERS = [LogisticRegression, RandomForestClassifier,
               GradientBoostingClassifier]


def _target(cls):
    return Y_REG if cls in REGRESSORS else Y_CLS


@pytest.mark.parametrize("cls", REGRESSORS + CLASSIFIERS)
def test_sklearn_backend_fits_with_defaults(cls):
    model = cls(backend='sklearn')
    assert model.backend == 'sklearn'
    model.fit(X, _target(cls))
    assert np.asarray(model.predict(X)).shape[0] == X.shape[0]


def test_kmeans_sklearn_backend_fits_with_defaults():
    model = KMeans(n_clusters=3, random_state=0, backend='sklearn')
    assert model.backend == 'sklearn'
    labels = model.fit(X).predict(X)
    assert np.asarray(labels).shape == (X.shape[0],)


@pytest.mark.parametrize("cls,kwargs", [
    (KMeans, {'n_clusters': 3, 'precompute_distances': True}),
    (KMeans, {'n_clusters': 3, 'n_jobs': 2}),
    (ElasticNet, {'normalize': True}),
    (Lasso, {'normalize': True}),
    (Ridge, {'normalize': True}),
    (LinearRegression, {'normalize': True}),
    (RandomForestClassifier, {'min_impurity_split': 0.1}),
    (RandomForestRegressor, {'min_impurity_split': 0.1}),
    (GradientBoostingClassifier, {'min_impurity_split': 0.1}),
    (GradientBoostingClassifier, {'presort': True}),
    (GradientBoostingRegressor, {'min_impurity_split': 0.1}),
    (GradientBoostingRegressor, {'presort': True}),
])
def test_removed_sklearn_params_are_accepted_and_warn(cls, kwargs):
    with pytest.warns(FutureWarning, match="ignored by the sklearn backend"):
        model = cls(backend='sklearn', **kwargs)
    model.fit(X, None if cls is KMeans else _target(cls))
    assert np.asarray(model.predict(X)).shape[0] == X.shape[0]


@pytest.mark.parametrize("cls,kwargs", [
    (KMeans, {'n_clusters': 3, 'algorithm': 'auto'}),
    (RandomForestRegressor, {'criterion': 'mae', 'max_features': 'auto'}),
    (RandomForestClassifier, {'max_features': 'auto'}),
    (GradientBoostingRegressor, {'loss': 'lad', 'criterion': 'mse'}),
    (GradientBoostingClassifier, {'loss': 'deviance',
                                  'max_features': 'auto'}),
])
def test_legacy_sklearn_values_are_translated(cls, kwargs):
    model = cls(backend='sklearn', **kwargs)
    model.fit(X, None if cls is KMeans else _target(cls))
    assert np.asarray(model.predict(X)).shape[0] == X.shape[0]


def test_auto_backend_routes_removed_param_to_sklearn():
    # A non-default sklearn-only parameter selects the sklearn backend; the
    # removed parameter itself is dropped with a warning instead of a
    # TypeError from the sklearn constructor.
    with pytest.warns(FutureWarning, match="'normalize' was removed"):
        model = ElasticNet(normalize=True)
    assert model.backend == 'sklearn'
    model.fit(X, Y_REG)
    assert np.asarray(model.predict(X)).shape[0] == X.shape[0]
