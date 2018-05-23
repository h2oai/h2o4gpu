# -*- encoding: utf-8 -*-
"""
KMeans solver tests using SKLearn datasets.

:copyright: 2017-2018 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import os
import time
from h2o4gpu import KMeans
from h2o4gpu.datasets import load_iris
import numpy as np
from h2o4gpu.datasets.samples_generator import make_blobs
from h2o4gpu.metrics.cluster import v_measure_score

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
            np.sort(model.cluster_centers_, axis=0), np.sort(model_rerun.cluster_centers_, axis=0)
        )

        model_rerun2 = model_rerun.fit(X)

        # Multiple invocations of fit with the same random_state
        # also should produce the same result
        assert np.allclose(
            np.sort(model_rerun.cluster_centers_, axis=0), np.sort(model_rerun2.cluster_centers_, axis=0)
        )

        model_all = KMeans(n_clusters=clusters, random_state=123).fit(X)

        # Multi GPU should yield same result as single GPU
        assert np.allclose(
            np.sort(model.cluster_centers_, axis=0), np.sort(model_all.cluster_centers_, axis=0)
        )

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

        print("Running model")
        model = KMeans(n_gpus=1, n_clusters=clusters, random_state=123).fit(X)

        assert model.cluster_centers_.shape == (X.shape[1], clusters)
        print("passed 1")

        print("Running model_rerun")
        model_rerun = KMeans(max_iter=1, n_gpus=1, n_clusters=clusters, random_state=123, init=model.cluster_centers_, n_init=1).fit(X)
        import sys
        print(model_rerun.cluster_centers_)
        sys.stdout.flush()

        # Choosing initial clusters for sklearn should yield similar result
        assert np.allclose(
           model.cluster_centers_, model_rerun.cluster_centers_
        )

        # sklearn directly or our indirect should be same (and is)
        from sklearn.cluster import KMeans as KMeans_test

        print("Running model_rerun2")
        model_rerun2 = KMeans_test(max_iter=1, n_clusters=clusters, random_state=123, init=model.cluster_centers_, n_init=1).fit(X)
        print(model_rerun2.cluster_centers_)
        sys.stdout.flush()

        assert np.allclose(
            model_rerun.cluster_centers_, model_rerun2.cluster_centers_
        )
        print("passed 2")

    def test_accuracy(self):
        from h2o4gpu.cluster import KMeansSklearn as skKMeans
        n_samples = 500000
        centers = 10
        X, true_labels = make_blobs(n_samples=n_samples, centers=centers,
                                    cluster_std=1., random_state=42)

        kmeans_h2o = KMeans(n_gpus=1, n_clusters=centers, random_state=42)
        kmeans_h2o.fit(X)
        kmeans_sk = skKMeans(n_init=1, n_clusters=centers, random_state=42)
        kmeans_sk.fit(X)

        accuracy_h2o = v_measure_score(kmeans_h2o.labels_, true_labels)
        accuracy_sk = v_measure_score(kmeans_sk.labels_, true_labels)
        # We also want to be either better or at most 10% worse than SKLearn
        # Everything else is horrible and we probably should fix something
        assert accuracy_h2o - accuracy_sk >= -0.1

    def test_speed_vs_sk(self):
        from h2o4gpu.cluster import KMeansSklearn as skKMeans
        n_samples = 100000
        centers = 10
        X, true_labels = make_blobs(n_samples=n_samples, centers=centers,
                                    cluster_std=1., random_state=42)

        kmeans_h2o = KMeans(n_gpus=1, n_clusters=centers, random_state=42)
        # Warmup - during first call CUDA kernels take ~2sec to load
        kmeans_h2o.fit(X)
        start_h2o = time.time()
        kmeans_h2o.fit(X)
        end_h2o = time.time()

        if os.getenv("CHECKPERFORMANCE") is not None:
            kmeans_sk = skKMeans(n_init=1, n_clusters=centers, algorithm='full', n_jobs=-1)
            start_sk = time.time()
            kmeans_sk.fit(X)
            end_sk = time.time()
            assert end_h2o - start_h2o <= end_sk - start_sk
