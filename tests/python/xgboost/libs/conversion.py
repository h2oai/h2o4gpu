import pandas as pd


def _get_nominal_integer_dict(nominal_vals):
    """Convert nominal values in integers, starting at 0.
    Parameters:
        nominal_vals (pd.Series): A series.
    Returns:
        d (dict): An dictionary with numeric values.

    """
    d = {}
    for val in nominal_vals:
        if val not in d:
            current_max = max(d.values()) if len(d) > 0 else -1
            d[val] = current_max+1
    return d


def _convert_to_integer(srs, d):
    """Convert series to integer, given a dictionary.
    Parameters:
        srs (pd.Series): A series.
        d (dict): A dictionary mapping values to integers
    Returns:
        srs (pd.Series): An series with numeric values.

    """
    return srs.map(lambda x: d[x])


def convert_cols_categorical_to_numeric(df, col_list=None):
    """Convert categorical columns to numeric and leave numeric columns
    as they are. You can force to convert a numerical column if it is
    included in col_list
    Parameters:
        df (pd.DataFrame): Dataframe.
        col_list (list): List of columns.
    Returns:
        ret (pd.DataFrame): An dataframe with numeric values.
    Examples:
        >>> df = pd.DataFrame({'letters':['a','b','c'],'numbers':[1,2,3]})
        >>> df_numeric = convert_cols_categorical_to_numeric(df)
        >>> print(df_numeric)
           letters  numbers
        0        0        1
        1        1        2
        2        2        3

    """
    if col_list is None: col_list = []
    ret = pd.DataFrame()
    for column_name in df.columns:
        column = df[column_name]
        if column.dtype == 'object' or column_name in col_list:
            col_dict = _get_nominal_integer_dict(column)
            ret[column_name] = _convert_to_integer(column, col_dict)
        else:
            ret[column_name] = column
    return ret


def convert_related_cols_categorical_to_numeric(df, col_list):
    """Convert categorical columns, that are related between each other,
    to numeric and leave numeric columns
    as they are.
    Parameters:
        df (pd.DataFrame): Dataframe.
        col_list (list): List of columns.
    Returns:
        ret (pd.DataFrame): An dataframe with numeric values.
    Examples:
        >>> df = pd.DataFrame({'letters':['a','b','c'],'letters2':['c','d','e'],'numbers':[1,2,3]})
        >>> df_numeric = convert_related_cols_categorical_to_numeric(df, col_list=['letters','letters2'])
        >>> print(df_numeric)
           letters  letters2  numbers
        0        0         2        1
        1        1         3        2
        2        2         4        3

    """
    ret = pd.DataFrame()
    values=None
    for c in col_list:
        values = pd.concat([values,df[c]], axis=0)
        values = pd.Series(values.unique())
    col_dict = _get_nominal_integer_dict(values)
    for column_name in df.columns:
        column = df[column_name]
        if column_name in col_list:
            ret[column_name] = _convert_to_integer(column, col_dict)
        else:
            ret[column_name] = column
    return ret

