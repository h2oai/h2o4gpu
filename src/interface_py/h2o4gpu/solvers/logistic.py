#- * - encoding : utf - 8 - * -
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
# pylint: disable=unused-import
from h2o4gpu.solvers import elastic_net
from h2o4gpu.linear_model import logistic

class LogisticRegression(elastic_net.GLM, logistic.LogisticRegression_sklearn):

    """H2O Logistic Regression Solver

    :param penalty : str, 'l1' or 'l2', default: 'l2'
        Used to specify the norm used in the penalization. The 'newton-cg',
        'sag' and 'lbfgs' solvers support only l2 penalties.
        .. versionadded:: 0.19
           l1 penalty with SAGA solver (allowing 'multinomial' + L1)

    :param dual : bool, default: False
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    :param tol : float, default: 1e-4
        Tolerance for stopping criteria.

    :param C : float, default: 1.0
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.

    :param fit_intercept : bool, default: True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

    :param intercept_scaling : float, default 1.
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

    :param class_weight : dict or 'balanced', default: None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.
        .. versionadded:: 0.17
           *class_weight='balanced'*

    :param random_state : int, RandomState instance or None, optional, default: None
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`. Used when ``solver`` == 'sag' or
        'liblinear'.

    :param solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'},
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
        preprocess the data with a scaler from sklearn.preprocessing.
        .. versionadded:: 0.17
           Stochastic Average Gradient descent solver.
        .. versionadded:: 0.19
           SAGA solver.

    :param max_iter : int, default: 100
        Useful only for the newton-cg, sag and lbfgs solvers.
        Maximum number of iterations taken for the solvers to converge.

    :param multi_class : str, {'ovr', 'multinomial'}, default: 'ovr'
        Multiclass option can be either 'ovr' or 'multinomial'. If the option
        chosen is 'ovr', then a binary problem is fit for each label. Else
        the loss minimised is the multinomial loss fit across
        the entire probability distribution. Does not work for liblinear
        solver.
        .. versionadded:: 0.18
           Stochastic Average Gradient descent solver for 'multinomial' case.

    :param verbose : int, default: 0
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    :param warm_start : bool, default: False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        Useless for liblinear solver.
        .. versionadded:: 0.17
           *warm_start* to support *lbfgs*, *newton-cg*, *sag*, *saga* solvers.

    :param n_jobs : int, default: 1
        Number of CPU cores used when parallelizing over classes if
        multi_class='ovr'". This parameter is ignored when the ``solver``is set
        to 'liblinear' regardless of whether 'multi_class' is specified or
        not. If given a value of -1, all cores are used.

    :param int n_gpus: Number of GPUs to use in GLM solver. Default is -1.

    :param bool glm_stop_early: Stop early when there is no more relative
        improvement in the primary and dual residuals for ADMM. Default is True

    :param float glm_stop_early_error_fraction: Relative tolerance for
        metric-based stopping criterion (stop if relative improvement is not at
        least this much). Default is 1.0.
    """

    def __init__(self,
                 penalty='l2', #h2o4gpu
                 dual=False,
                 tol=1E-2, #h2o4gpu
                 C=1.0, #h2o4gpu
                 fit_intercept=True, #h2o4gpu
                 intercept_scaling=1,
                 class_weight=None,
                 random_state=None,
                 solver='liblinear',
                 max_iter=5000, #h2o4gpu
                 multi_class='ovr',
                 verbose=0, #h2o4gpu
                 warm_start=False,
                 n_jobs=1,
                 n_gpus=-1, #h2o4gpu
                 glm_stop_early=True, #h2o4gpu
                 glm_stop_early_error_fraction=1.0): #h2o4gpu

        # setup backup to sklearn class
        # (can remove if fully implement sklearn functionality)
        do_sklearn = False

        if dual != False or intercept_scaling != 1 or class_weight != None or random_state != None \
                or solver != 'liblinear' or multi_class != 'ovr' or warm_start != False or n_jobs != 1:
                do_sklearn = True

        if do_sklearn:
            print("USING SKLEARN")
            super(LogisticRegression,self).__init__(penalty=penalty,
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
        else:
            #TODO type checking!
            n_threads = None
            n_alphas = 1 #n_alphas is always 1 since we are only dealing with l1 or l2 regularization
            n_lambdas = 1 #n_lambdas is always 1 since we are only dealing with l1 or l2 regularization
            n_folds = 1
            lambda_max = 1.0/C
            lambda_min_ratio = 1.0
            lambda_stop_early = False
            store_full_path = 0
            alphas = None
            lambdas = None

            if penalty is 'l2':
                alpha_min = 0.0
                alpha_max = 0.0
            elif penalty is 'l1':
                alpha_min = 1.0
                alpha_max = 1.0

            super(LogisticRegression, self).__init__(
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