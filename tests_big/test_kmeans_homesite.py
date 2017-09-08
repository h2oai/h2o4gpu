# -*- encoding: utf-8 -*-
"""
KMeans solver tests using Kaggle datasets.

:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import h2o4gpu as h2o4gpu
import pandas as pd
import numpy as np


class TestKmeans(object):
    def test_preds_vs_scikit(self):
        X = self._fetch_data()
        model = self._train_model(X)

        train_labels = model.predict(X)
        print(train_labels)
        sklearn_labels = model.sklearn_predict(X)
        print(sklearn_labels)

        diffs = 0
        for tl in zip(train_labels, sklearn_labels):
            if tl[0] != tl[1]:
                diffs = diffs + 1

        assert diffs / X.shape[0] <= 0.1

    def test_transform_vs_scikit(self):
        X = self._fetch_data()
        model = self._train_model(X)

        h2o_labels = \
            list(map(lambda x: np.argmin(x), model.transform(X)))

        sklearn_labels = \
            list(
                map(lambda x: np.argmin(x),
                    model.sklearn_transform(X))
            )

        diffs = 0
        for tl in zip(h2o_labels, sklearn_labels):
            if tl[0] != tl[1]:
                diffs = diffs + 1

        assert diffs / X.shape[0] <= 0.1

    @staticmethod
    def _train_model(trainencflt):
        model = h2o4gpu.KMeans(n_gpus=1, n_clusters=10, tol=1e-7,
                               max_iter=100, init_data='selectstrat')
        return model.fit(trainencflt)

    @staticmethod
    def _fetch_data():
        import pandas as pd
        if 1==1:
            from h2o4gpu.datasets import fetch_20newsgroups, fetch_20newsgroups_vectorized, fetch_california_housing, \
                fetch_covtype, fetch_kddcup99, fetch_lfw_pairs, fetch_lfw_people, fetch_mldata, fetch_olivetti_faces, \
                fetch_rcv1, fetch_species_distributions
            from h2o4gpu.model_selection import train_test_split
            data = fetch_covtype()
            sizetokeep = 1000  # 1k rows for now
            sizetokeep = min(sizetokeep, len(data.data[:, 0]))
            X = data.data[0:sizetokeep, :]
            y = data.target[0:sizetokeep]
            print("Got Data")

            # Create 0.8/0.2 train/test split
            print("Split Data")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8,
                                                                random_state=42)

            test= pd.DataFrame(X_test)
            train = pd.DataFrame(X_train)

        # https://github.com/caesar0301/awesome-public-datasets
        else:
            # TODO: FIXME
            train_file = "./data/???_train.csv"
            test_file = "./data/???_test.csv"

            train = pd.read_csv(train_file)
            test = pd.read_csv(test_file)
            train.drop(['QuoteConversion_Flag'], axis=1, inplace=True)

        dataset = pd.concat([train, test], ignore_index=True)
        tmp = dataset.dtypes.reset_index().rename(
            columns={0: "type"})  # ["index"]
        indx = tmp["type"] == "object"
        categoricals = tmp[indx]["index"].tolist()
        # replace nans as new category
        for col in dataset.columns:
            dataset[col] = dataset[col].fillna("__NA__")

        # encode unfreq categories
        for col in categoricals:
            val_dict = dataset[col].value_counts()
            val_dict = dataset[col].value_counts().reset_index()
            indx = val_dict[col] < 100
            res = val_dict[indx]["index"].tolist()
            indx = dataset[col].isin(res)
            vals = dataset[col].values
            vals[indx] = "___UNFREQ___"
            dataset[col] = vals

        # encode all as freqs
        for col in categoricals:
            val_dict = dataset[col].value_counts()
            val_dict = val_dict / float(dataset.shape[0])
            val_dict = val_dict.to_dict()
            dataset[col] = dataset[col].apply(lambda x: val_dict[x])

        # replace nans as new category
        for col in dataset.columns:
            dataset[col] = dataset[col].replace("__NA__", 0)
        trainenc = dataset.iloc[:train.shape[0], :].reset_index(drop=True)
        trainencflt = trainenc.values.astype(np.float32)
        return trainencflt
