import sys
import feather
import numpy as np
import pandas as pd

def import_data(data_path, use_pandas=False, intercept=True,valid_fraction=0.2,classification=True):
    #Can import data using pandas or feather.
    use_pandas =use_pandas

    data_file = data_path #If importing using pandas

    if use_pandas:
        print("Reading Data with Pandas")
        data = pd.read_csv(data_file)
    else:
        print("Reading Data with Feather")
        data = feather.read_dataframe(data_file)
    print(data.shape)
    data_x = np.array(data.iloc[:, :data.shape[1] - 1], dtype='float32', order='C')
    data_y = np.array(data.iloc[:, data.shape[1] - 1], dtype='float32', order='C')

    #Setup train/validation set split (assuming form of mxn where m=row count and n=col count)
    morig = data_x.shape[0]
    norig = data_x.shape[1]
    print("Original m=%d n=%d" % (morig, norig))
    sys.stdout.flush()

    #Do train/valid split
    if valid_fraction > 0:
        valid_fraction = valid_fraction
        HO = int(valid_fraction * morig)
        H = morig - HO
        print("Size of Train rows=%d & valid rows=%d" % (H, HO))
        sys.stdout.flush()
        train_x = np.copy(data_x[0:H, :])
        train_y = np.copy(data_y[0:H])
        valid_x = np.copy(data_x[H:morig, :])
        valid_y = np.copy(data_y[H:morig])
        print("Size of Train cols=%d valid cols=%d" % (train_x.shape[1], valid_x.shape[1]))
    else:
        train_x = data_x
        train_y = data_y

    #Using intercept
    if intercept:
        train_x = np.hstack([train_x, np.ones((train_x.shape[0], 1), dtype=train_x.dtype)])
        if valid_fraction > 0:
            valid_x = np.hstack([valid_x, np.ones((valid_x.shape[0], 1), dtype=valid_x.dtype)])
            print("Size of Train cols=%d & valid cols=%d after adding intercept column" % (train_x.shape[1], valid_x.shape[1]))
        else:
            print("Size of Train cols=%d after adding intercept column" % (train_x.shape[1]))

    if classification:
        family="logistic"
    else:
        family="elasticnet"
    if valid_fraction > 0:
        return train_x,train_y,valid_x,valid_y,family
    else:
        return train_x,train_y,family