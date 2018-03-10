#- * - encoding : utf - 8 - * -
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""


def import_data(data_path,
                use_pandas=False,
                intercept=True,
                valid_fraction=0.2,
                classification=True):
    """Import Data for H2O GPU Edition

    This function will read in data and prepare it for H2O4GPU's GLM solver.

    Note, the data is assumed to be all numeric,i.e.,
    categoricals are one hot encoded, etc.

    :param data_path : str
                 A path to a dataset (The dataset needs to be all numeric)
    :param use_pandas : bool
                  Indicate if Pandas should be used to parse
    :param intercept : bool
                  Indicate if intercept term is needed
    :param valid_fraction : float
                      Percentage of dataset reserved for a validation set
    :param classification : bool
                      Classification problem?
    :returns
    If valid_fraction > 0 it will return the following:
        train_x: numpy array of train input variables
        train_y: numpy array of y variable
        valid_x: numpy array of valid input variables
        valid_y: numpy array of valid y variable
        family : string that would either be "logistic" if classification is set
            to True, otherwise "elasticnet"
    If valid_fraction == 0 it will return the following:
        train_x: numpy array of train input variables
        train_y: numpy array of y variable
        family : string that would either be "logistic" if classification is set
            to True, otherwise "elasticnet"
    """
    #Can import data using pandas or feather.
    use_pandas = use_pandas

    data_file = data_path  # If importing using pandas

    if use_pandas:
        print("Reading Data with Pandas")
        import pandas as pd
        data = pd.read_csv(data_file)
    else:
        print("Reading Data with Feather")
        import feather
        data = feather.read_dataframe(data_file)
    print(data.shape)
    import numpy as np
    data_x = np.array(
        data.iloc[:, :data.shape[1] - 1],
        dtype='float32',
        order='C',
        copy=False)
    data_y = np.array(
        data.iloc[:, data.shape[1] - 1], dtype='float32', order='C', copy=False)

    #Setup train / validation set split
    #(assuming form of mxn where m = row count and n = col count)
    morig = data_x.shape[0]
    norig = data_x.shape[1]
    print("Original m=%d n=%d" % (morig, norig))
    import sys
    sys.stdout.flush()

    #Do train / valid split
    if valid_fraction > 0:
        valid_fraction = valid_fraction
        HO = int(valid_fraction * morig)
        H = morig - HO
        print("Size of Train rows=%d & valid rows=%d" % (H, HO))
        sys.stdout.flush()
        train_x = data_x[0:H, :]
        train_y = data_y[0:H]
        valid_x = data_x[H:morig, :]
        valid_y = data_y[H:morig]
        print("Size of Train cols=%d valid cols=%d" % (train_x.shape[1],
                                                       valid_x.shape[1]))
    else:
        train_x = data_x
        train_y = data_y


#Using intercept
    if intercept:
        train_x = np.hstack(
            [train_x,
             np.ones((train_x.shape[0], 1), dtype=train_x.dtype)])
        if valid_fraction > 0:
            valid_x = np.hstack(
                [valid_x,
                 np.ones((valid_x.shape[0], 1), dtype=valid_x.dtype)])
            print("Size of Train cols=%d & valid cols=%d after adding "
                  "intercept column" % (train_x.shape[1], valid_x.shape[1]))
        else:
            print("Size of Train cols=%d after adding intercept column" %
                  (train_x.shape[1]))

    if classification:
        family = "logistic"
    else:
        family = "elasticnet"
    if valid_fraction > 0:
        return train_x, train_y, valid_x, valid_y, family

    return train_x, train_y, family
