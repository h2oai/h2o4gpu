import pandas as pd
import numpy as np
import h2ogpuml as h2ogpuml
from os.path import expanduser

home = expanduser("~")
train_file = home + "/Dev/code/h2ogpuml-benchmarks/Data/Homesite/train.csv"
test_file = home + "/Dev/code/h2ogpuml-benchmarks/Data/Homesite/test.csv"

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

train.drop(['QuoteConversion_Flag'], axis=1, inplace=True)

dataset = pd.concat([train, test], ignore_index=True)
tmp = dataset.dtypes.reset_index().rename(columns={0: "type"})  # ["index"]
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
testenc = dataset.iloc[train.shape[0]:, :].reset_index(drop=True)

trainencflt = trainenc.values.astype(np.float32)
testencflt = testenc.values.astype(np.float32)
k = 1000
rows = np.shape(trainencflt)[0]
print(rows)
np.random.seed(1234)
# labels = np.random.randint(rows, size=rows) % k
import random
import numpy as np

labels = np.asarray([])
num = int(rows / k)
for x in range(0, num + 1):
    if x < num:
        many = k
    else:
        many = rows % k
    labels = np.append(labels, np.asarray(random.sample(range(k), many)))
print(labels.shape)

n_gpus = 1
model = h2ogpuml.KMeansGPU(n_gpus=n_gpus, k=k, threshold=1e-7, max_iterations=1000)
centroids, timefit0, timefit1 = model.fit(trainencflt, labels)

# train_labels = model.predict(trainencflt)
