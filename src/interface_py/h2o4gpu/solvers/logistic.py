# - * - encoding : utf - 8 - * -
# pylint: disable=fixme, line-too-long
"""
:copyright: 2017-2018 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
# pylint: disable=unused-import
import numpy as np
from h2o4gpu.solvers import elastic_net
from h2o4gpu.linear_model import logistic as sk
from ..solvers.utils import _setter


class LogisticRegression(object):
    """H2O Logistic Regression Solver

    Parameters
    ----------
    penalty : str, 'l1' or 'l2', default: 'l2'
        Used to specify the norm used in the penalization. The 'newton-cg',
        'sag' and 'lbfgs' solvers support only l2 penalties.

        .. versionadded:: 0.19
           l1 penalty with SAGA solver (allowing 'multinomial' + L1)

    dual : bool, default: False
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    tol : float, default: 1e-4
        Tolerance for stopping criteria.

    C : float, default: 1.0
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.

    fit_intercept : bool, default: True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

    intercept_scaling : float, default 1.
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equal to
        intercept_scaling is appended to the instance vector.
        The intercept becomes ``intercept_scaling * synthetic_feature_weight``.

        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    class_weight : dict or 'balanced', default: None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

        .. versionadded:: 0.17
           *class_weight='balanced'*

    random_state : int, RandomState instance or None, optional, default: None
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`. Used when ``solver`` == 'sag' or
        'liblinear'.

    solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'},
        default: 'liblinear'
        Algorithm to use in the optimization problem.

        - For small datasets, 'liblinear' is a good choice, whereas 'sag' and
            'saga' are faster for large ones.
        - For multiclass problems, only 'newton-cg', 'sag', 'saga' and 'lbfgs'
            handle multinomial loss; 'liblinear' is limited to one-versus-rest
            schemes.
        - 'newton-cg', 'lbfgs' and 'sag' only handle L2 penalty, whereas
            'liblinear' and 'saga' handle L1 penalty.

        Note that 'sag' and 'saga' fast convergence is only guaranteed on
        features with approximately the same scale. You can
        preprocess the data with a scaler from h2o4gpu.preprocessing.

        .. versionadded:: 0.17
           Stochastic Average Gradient descent solver.
        .. versionadded:: 0.19
           SAGA solver.

    max_iter : int, default: 100
        Useful only for the newton-cg, sag and lbfgs solvers.
        Maximum number of iterations taken for the solvers to converge.

    multi_class : str, {'ovr', 'multinomial'}, default: 'ovr'
        Multiclass option can be either 'ovr' or 'multinomial'. If the option
        chosen is 'ovr', then a binary problem is fit for each label. Else
        the loss minimised is the multinomial loss fit across
        the entire probability distribution. Does not work for liblinear
        solver.

        .. versionadded:: 0.18
           Stochastic Average Gradient descent solver for 'multinomial' case.

    verbose : int, (Default=0)
       Print verbose information to the console if set to > 0.

    warm_start : bool, default: False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        Useless for liblinear solver.

        .. versionadded:: 0.17
           *warm_start* to support *lbfgs*, *newton-cg*, *sag*, *saga* solvers.

    n_jobs : int, default: 1
        Number of CPU cores used when parallelizing over classes if
        multi_class='ovr'". This parameter is ignored when the ``solver``is set
        to 'liblinear' regardless of whether 'multi_class' is specified or
        not. If given a value of -1, all cores are used.

    n_gpus : int
        Number of gpu's to use in RandomForestRegressor solver. Default is -1.

    glm_stop_early : bool, (Default=True)
       Stop early when there is no more relative
       improvement in the primary and dual residuals for ADMM.

    glm_stop_early_error_fraction : float, (Default=1.0)
       Relative tolerance for metric-based stopping criterion (stop if relative improvement is not at
       least this much).

    backend : string, (Default="auto")
        Which backend to use.
        Options are 'auto', 'sklearn', 'h2o4gpu'.
        Saves as attribute for actual backend used.
    """

    def __init__(
            self,
            penalty='l2',  # h2o4gpu
            dual=False,
            tol=1E-2,  # h2o4gpu
            C=1.0,  # h2o4gpu
            fit_intercept=True,  # h2o4gpu
            intercept_scaling=1.0,
            class_weight=None,
            random_state=None,
            solver='liblinear',
            max_iter=5000,  # h2o4gpu
            multi_class='ovr',
            verbose=0,  # h2o4gpu
            warm_start=False,
            n_jobs=1,
            n_gpus=-1,  # h2o4gpu
            glm_stop_early=True,  # h2o4gpu
            glm_stop_early_error_fraction=1.0,
            backend='auto'):  # h2o4gpu
        import os
        _backend = os.environ.get('H2O4GPU_BACKEND', None)
        if _backend is not None:
            backend = _backend

        # Fall back to Sklearn
        # Can remove if fully implement sklearn functionality
        self.do_sklearn = False
        if backend == 'auto':
            params_string = [
                'intercept_scaling', 'class_weight', 'solver', 'multi_class'
            ]
            params = [intercept_scaling, class_weight, solver, multi_class]
            params_default = [1.0, None, 'liblinear', 'ovr']

            i = 0
            for param in params:
                if param != params_default[i]:
                    self.do_sklearn = True
                    if verbose:
                        print("WARNING:"
                              " The sklearn parameter " + params_string[i] +
                              " has been changed from default to " + str(param)
                              + "  Will run Sklearn Logistic Regression.")
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

        self.model_sklearn = sk.LogisticRegressionSklearn(
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs)

        # Equivalent Logistic parameters for h2o4gpu
        n_threads = None
        n_alphas = 1
        n_lambdas = 1
        n_folds = 1
        lambda_max = 1.0 / C
        lambda_min_ratio = 1.0
        lambda_stop_early = False
        store_full_path = 0
        alphas = None
        lambdas = None

        # Utilize penalty parameter to setup alphas
        if penalty == 'l2':
            alpha_min = 0.0
            alpha_max = 0.0
        elif penalty == 'l1':
            alpha_min = 1.0
            alpha_max = 1.0
        else:
            assert ValueError, "penalty should be either l1 " \
                               "or l2 but got " + penalty

        self.model_h2o4gpu = elastic_net.ElasticNetH2O(
            n_threads=n_threads,
            n_gpus=n_gpus,
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
            family='logistic',
            store_full_path=store_full_path,
            lambda_max=lambda_max,
            alpha_max=alpha_max,
            alpha_min=alpha_min,
            alphas=alphas,
            lambdas=lambdas,
            order=None)

        if self.do_sklearn:
            if verbose:
                print("Running sklearn Logistic Regression")
            self.model = self.model_sklearn
        else:
            if verbose:
                print("Running h2o4gpu Logistic Regression")
            self.model = self.model_h2o4gpu
        self.verbose = verbose

    def fit(self, X, y=None, sample_weight=None):
        res = self.model.fit(X, y, sample_weight)
        self.set_attributes()
        return res

    def predict_proba(self, X):
        if self.do_sklearn:
            res = self.model.predict_proba(X)
            self.set_attributes()
            return res
        res = self.model.predict(X)
        self.set_attributes()
        return res

    def decision_function(self, X):
        if self.verbose:
            print("WARNING: decision_function() is using sklearn")
        return self.model_sklearn.decision_function(X)

    def densify(self):
        if self.do_sklearn:
            return self.model.densify()
        return None

    def get_params(self):
        return self.model.get_params()

    def predict(self, X):
        if self.do_sklearn:
            res = self.model.predict(X)
            self.set_attributes()
            return res
        res = self.model.predict(X)
        res[res < 0.5] = 0
        res[res > 0.5] = 1
        self.set_attributes()
        return res.squeeze()

    def predict_log_proba(self, X):
        res = self.predict_proba(X)
        self.set_attributes()
        return np.log(res)

    def score(self, X, y, sample_weight=None):
        # TODO add for h2o4gpu
        if self.verbose:
            print("WARNING: score() is using sklearn")
        if not self.do_sklearn:
            self.model_sklearn.fit(X, y)  # Need to re-fit
        res = self.model_sklearn.score(X, y, sample_weight)
        return res

    def set_params(self, **params):
        return self.model.set_params(**params)

    def sparsify(self):
        if self.do_sklearn:
            return self.model.sparsify()
        assert ValueError, "sparsify() is not yet supporte for h2o4gpu"
        return None

    def set_attributes(self):
        """ set attributes for Logistic
        """
        s = _setter(oself=self, e1=NameError, e2=AttributeError)

        s('oself.coef_ = oself.model.coef_')
        s('oself.intercept_ = oself.model.intercept_')
        s('oself.n_iter_ = oself.model.n_iter_')

        self.time_prepare = None
        s('oself.time_prepare = oself.model.time_prepare')
        self.time_upload_data = None
        s('oself.time_upload_data = oself.model.time_upload_data')
        self.time_fitonly = None
        s('oself.time_fitonly = oself.model.time_fitonly')
