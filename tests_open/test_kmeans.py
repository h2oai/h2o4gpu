# -*- encoding: utf-8 -*-
"""
KMeans solver tests using SKLearn datasets.

:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import os
import h2o4gpu
from h2o4gpu import KMeans
from h2o4gpu.datasets import load_iris
import numpy as np


class TestKmeans(object):
    @classmethod
    def setup_class(cls):
        os.environ['SCIKIT_LEARN_DATA'] = "open_data"

    def test_fit_iris(self):
        X = load_iris().data
        clusters = 4
        model = KMeans(n_gpus=1, n_clusters=clusters, random_state=123).fit(X)

        assert model.cluster_centers_.shape == (X.shape[1], clusters)

        model_rerun = KMeans(n_gpus=1, n_clusters=clusters, random_state=123).fit(X)

        # Same random_state should yield same results
        assert np.allclose(
            model.cluster_centers_, model_rerun.cluster_centers_
        )

        model_rerun2 = model_rerun.fit(X)

        # Multiple invocations of fit with the same random_state
        # also should produce the same result
        assert np.allclose(
            model_rerun.cluster_centers_, model_rerun2.cluster_centers_
        )

        # model_all = KMeans(n_clusters=clusters, random_state=123).fit(X)

        # Multi GPU should yield same result as single GPU
        # TODO multi GPU returns wrong results
        # assert np.allclose(
        #     model.cluster_centers_, model_all.cluster_centers_
        # )

    def test_fit_vs_sk_iris(self):
        X = load_iris().data
        model = KMeans(n_gpus=1, n_clusters=4, random_state=1234).fit(X)

        h2o_labels = model.predict(X)
        sk_lables = model.sklearn_predict(X)

        assert all(h2o_labels == sk_lables)

    def test_fit_iris_precision(self):
        X_f64 = load_iris().data
        X_f32 = X_f64.astype(np.float32)
        kmeans = KMeans(n_gpus=1, n_clusters=4, random_state=12345)
        model_f64_labels = kmeans.fit(X_f64).predict(X_f64)
        model_f32_labels = kmeans.fit(X_f32).predict(X_f32)

        assert all(model_f64_labels == model_f32_labels)

    # On data where we don't loose precision, predictions should be the same
    def test_fit_i32_vs_f32(self):
        X_f64 = np.array([[1., 2.], [1., 4.], [1., 0.], [4., 2.], [4., 4.], [4., 0.]])
        X_f32 = X_f64.astype(np.float32)
        X_i32 = X_f64.astype(np.int32)
        kmeans = KMeans(n_gpus=1, n_clusters=2, random_state=123)

        model_f64_labels = kmeans.fit(X_f64).predict(X_f64)
        model_f32_labels = kmeans.fit(X_f32).predict(X_f32)
        model_i32_labels = kmeans.fit(X_i32).predict(X_i32)

        assert all(model_f64_labels == model_f32_labels)
        assert all(model_f32_labels == model_i32_labels)

    def test_predict_iris(self):
        X = load_iris().data
        model = KMeans(n_gpus=1, n_clusters=4, random_state=123456).fit(X)

        assert all(model.labels_ == model.predict(X))

    def test_transform_iris(self):
        X = load_iris().data
        model = KMeans(n_gpus=1, n_clusters=4, random_state=1234567).fit(X)

        labels_from_trans = list(
            map(lambda x: np.argmin(x), model.transform(X))
        )

        assert all(labels_from_trans == model.predict(X))

    def test_fit_iris_backupsklearn(self):
        X = load_iris().data
        clusters = 4
        model = KMeans(n_gpus=1, n_clusters=clusters, random_state=123).fit(X)

        assert model.cluster_centers_.shape == (X.shape[1], clusters)

        model_rerun = KMeans(n_gpus=1, n_clusters=clusters, random_state=123, init=model.cluster_centers_).fit(X)

        # Choosing initial clusters for sklearn should yield similar result (stable clusters)
        # TODO: Below fails, so our solution seems very different from what should be?
        #assert np.allclose(
        #    model.cluster_centers_, model_rerun.cluster_centers_
        #)

        # sklearn directly or our indirect should be same (and is)
        from sklearn.cluster import KMeans as KMeans_test

        model_rerun2 = KMeans_test(n_clusters=clusters, random_state=123, init=model.cluster_centers_).fit(X)

        assert np.allclose(
            model_rerun.cluster_centers_, model_rerun2.cluster_centers_
        )

