import h2o4gpu as h2o4gpu
import sys
import pandas as pd
import numpy as np
import random


class TestKmeans(object):

    def test_preds_vs_scikit(self):
        trainencflt = self._fetch_data()
        model, labels = self._train_model(trainencflt)

        train_labels = model.predict(trainencflt)
        sklearn_labels = model.sklearn_predict(trainencflt, labels)

        diffs = 0
        for tl in zip(train_labels, sklearn_labels):
            if tl[0] != tl[1]:
                diffs = diffs + 1

        assert diffs/trainencflt.shape[0] <= 0.1

    def test_transform_vs_scikit(self):
        trainencflt = self._fetch_data()
        model, labels = self._train_model(trainencflt)

        train_labels = model.transform(trainencflt)
        sklearn_labels = model.sklearn_transform(trainencflt, labels)

        diffs = 0
        for tl in zip(train_labels, sklearn_labels):
            if np.array_equal(tl[0], tl[1]):
                diffs = diffs + 1

        assert diffs/trainencflt.shape[0] <= 0.1

    @staticmethod
    def _train_model(trainencflt):
        k = 10
        rows = np.shape(trainencflt)[0]
        print(rows)
        np.random.seed(1234)
        # labels = np.random.randint(rows, size=rows) % k
        labels = np.asarray([])
        num = int(rows / k)
        for x in range(0, num + 1):
            if x < num:
                many = k
            else:
                many = rows % k
            labels = np.append(labels,
                               np.asarray(random.sample(range(k), many)))
        print(labels.shape)
        n_gpus = 1
        model = h2o4gpu.KMeans(n_gpus=n_gpus, n_clusters=k, tol=1e-7,
                                max_iter=100)
        model.fit(trainencflt, labels)
        return model, labels

    @staticmethod
    def _fetch_data():
        train_file = "./data/kmeans_homesite_train.csv"
        test_file = "./data/kmeans_homesite_test.csv"

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
