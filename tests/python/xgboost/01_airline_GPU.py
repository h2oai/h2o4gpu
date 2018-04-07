# coding: utf-8
"""
:copyright: 2017-2018 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
# # Experiment 01: Airline dataset (GPU version)
# 
# In this experiment we use [the airline dataset](http://kt.ijs.si/elena_ikonomovska/data.html) to predict arrival delay. The dataset consists of a large amount of records, containing flight arrival and departure details for all the commercial flights within the USA, from October 1987 to April 2008. Its size is around 116 million records and 5.76 GB of memory.
# 
# The details of the machine we used and the version of the libraries can be found in [experiment 01](01_airline.ipynb).

# In[1]:


import os,sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from libs.metrics import classification_metrics_binary, classification_metrics_binary_prob, binarize_prediction
from libs.loaders import load_airline
from libs.conversion import convert_cols_categorical_to_numeric, convert_related_cols_categorical_to_numeric
from libs.timer import Timer
from libs.utils import get_number_processors
import pkg_resources
import json
import warnings

print("System version: {}".format(sys.version))
print("XGBoost version: {}".format(pkg_resources.get_distribution('xgboost').version))
print("LightGBM version: {}".format(pkg_resources.get_distribution('lightgbm').version))

warnings.filterwarnings("ignore")


# # 1) XGBoost vs LightGBM benchmark
# In the next section we compare both libraries speed, accuracy and other metrics for the dataset of airline arrival delay. 
# 
# ### Data loading and management

# In[2]:


get_ipython().run_cell_magic('time', '', 'df_plane = load_airline()\nprint(df_plane.shape)')


# In[3]:


df_plane.head()


# The first step is to convert the categorical features to numeric features.

# In[4]:


get_ipython().run_cell_magic('time', '', "df_plane_numeric = convert_related_cols_categorical_to_numeric(df_plane, col_list=['Origin','Dest'])\ndel df_plane")


# In[5]:


df_plane_numeric.head()


# In[6]:


get_ipython().run_cell_magic('time', '', "df_plane_numeric = convert_cols_categorical_to_numeric(df_plane_numeric, col_list='UniqueCarrier')")


# In[7]:


df_plane_numeric.head()


# To simplify the pipeline, we are going to set a classification problem where the goal is to classify wheather a flight has arrived delayed or not. For that we need to binarize the variable `ArrDelay`.
# 
# If you want to extend this experiment, you can set a regression problem and try to identify the number of minutes of delay a fight has. Both XGBoost and LightGBM have regression classes.

# In[8]:


df_plane_numeric = df_plane_numeric.apply(lambda x: x.astype('int16'))


# In[9]:


get_ipython().run_cell_magic('time', '', "df_plane_numeric['ArrDelayBinary'] = 1*(df_plane_numeric['ArrDelay'] > 0)")


# In[10]:


df_plane_numeric.head()


# Once the features are prepared, let's split the dataset into train, validation and test set. We won't use validation for this example (however, you can try to add it).

# In[11]:


def split_train_val_test_df(df, val_size=0.2, test_size=0.2):
    train, validate, test = np.split(df.sample(frac=1), 
                                     [int((1-val_size-test_size)*len(df)), int((1-test_size)*len(df))])
    return train, validate, test


# In[12]:


get_ipython().run_cell_magic('time', '', 'train, validate, test = split_train_val_test_df(df_plane_numeric, val_size=0, test_size=0.2)\n\n#smaller dataset depending on the size of your GPU\n#df_plane_numeric_small = df_plane_numeric.sample(n=1e6).reset_index(drop=True)\n#train, validate, test = split_train_val_test_df(df_plane_numeric_small, val_size=0, test_size=0.2)\n\nprint(train.shape)\nprint(validate.shape)\nprint(test.shape)')


# In[13]:


def generate_feables(df):
    X = df[df.columns.difference(['ArrDelay', 'ArrDelayBinary'])]
    y = df['ArrDelayBinary']
    return X,y


# In[14]:


get_ipython().run_cell_magic('time', '', 'X_train, y_train = generate_feables(train)\nX_test, y_test = generate_feables(test)')


# ### Training and Evaluation
# Now we are going to create two pipelines, one of XGBoost and one for LightGBM. The technology behind both libraries is different, so it is difficult to compare them in the exact same model setting. XGBoost grows the trees depth-wise and controls model complexity with `max_depth`. Instead, LightGBM uses a leaf-wise algorithm and controls the model complexity by `num_leaves`. As a tradeoff, we use XGBoost with `max_depth=8`, which will have max number leaves of 255, and compare it with LightGBM with `num_leaves=255`. 

# In[15]:


results_dict = dict()
num_rounds = 200


# In[16]:


def train_xgboost(parameters, X, y, num_rounds=50):
    ddata = xgb.DMatrix(data=X, label=y, nthread=-1)
    with Timer() as t:
        clf = xgb.train(parameters, ddata, num_boost_round=num_rounds)
    return clf, t.interval


# In[17]:


def test_xgboost(clf, X, y):
    ddata = xgb.DMatrix(data=X, label=y, nthread=-1)
    with Timer() as t:
        y_pred = clf.predict(ddata)
    return y_pred, t.interval


# Let's start with the XGBoost model.

# In[18]:


xgb_params = {'max_depth':8, #'max_depth':2, 
              'objective':'binary:logistic', 
              'min_child_weight':30, 
              'eta':0.1, 
              'scale_pos_weight':2, 
              'gamma':0.1, 
              'reg_lamda':1, 
              'subsample':1,
              'tree_method':'gpu_hist' # exact can't handle airlines, so just use hist for now
             }


# *NOTE: We got an out of memory error with xgb due to the big size of the dataset and the tree. At the end of the notebook we perform a benchmark with different data sizes.*

# ```python
# xgb_clf_pipeline, t_train = train_xgboost(xgb_params, X_train, y_train, num_rounds)
# 
# results_dict['xgb']={ 'train_time': t_train }
# 
# y_prob_xgb, t_test = test_xgboost(xgb_clf_pipeline, X_test, y_test)
# 
# y_pred_xgb = binarize_prediction(y_prob_xgb)
# 
# results_dict['xgb']['test_time'] = t_test
# 
# del xgb_clf_pipeline
# ```
# 

# Training XGBoost model with leaf-wise growth

# In[19]:


xgb_hist_params = {'max_depth':8, 
                  'objective':'binary:logistic', 
                  'min_child_weight':30, 
                  'eta':0.1, 
                  'scale_pos_weight':2, 
                  'gamma':0.1, 
                  'reg_lamda':1, 
                  'subsample':1,
                  'tree_method':'gpu_hist'
                 }


# In[20]:


xgb_hist_clf_pipeline, t_train = train_xgboost(xgb_hist_params, X_train, y_train, num_rounds)


# In[21]:


results_dict['xgb_hist']={
    'train_time': t_train
}


# In[22]:


y_prob_xgb_hist, t_test = test_xgboost(xgb_hist_clf_pipeline, X_test, y_test)


# In[23]:


y_pred_xgb_hist = binarize_prediction(y_prob_xgb_hist)


# In[24]:


results_dict['xgb_hist']['test_time'] = t_test


# In[25]:


del xgb_hist_clf_pipeline


# Training LightGBM model

# In[26]:


def train_lightgbm(parameters, X, y, num_rounds=50):
    ddata = lgb.Dataset(X.values, y.values, free_raw_data=False)
    with Timer() as t:
        clf = lgb.train(parameters, ddata, num_boost_round=num_rounds)
    return clf, t.interval


# In[27]:


def test_lightgbm(clf, X):
    with Timer() as t:
        y_pred = clf.predict(X.values)
    return y_pred, t.interval


# In[28]:


lgbm_params = {'num_leaves': 2**8,
               'learning_rate': 0.1,
               'scale_pos_weight': 2,
               'min_split_gain': 0.1,
               'min_child_weight': 30,
               'reg_lambda': 1,
               'subsample': 1,
               'objective':'binary',
               'device': 'gpu',
               'task': 'train'
              }


# In[29]:


lgbm_clf_pipeline, t_train = train_lightgbm(lgbm_params, X_train, y_train, num_rounds)


# In[30]:


results_dict['lgbm']={
    'train_time': t_train
}


# In[31]:


y_prob_lgbm, t_test = test_lightgbm(lgbm_clf_pipeline, X_test)


# In[32]:


y_pred_lgbm = binarize_prediction(y_prob_lgbm)


# In[33]:


results_dict['lgbm']['test_time'] = t_test


# In[34]:


del lgbm_clf_pipeline


# As it can be seen in the results, given the specific versions and parameters of both XGBoost and LightGBM and in this specific dataset, LightGBM is faster. 
# 
# In general terms, leaf-wise algorithms are more efficient, they converge much faster than depth-wise. However, it may cause over-fitting when the data is small or there are too many leaves.

# ### Metrics
# We are going to obtain some metrics to evaluate the performance of each of the models.

# ```python
# report_xgb = classification_metrics_binary(y_test, y_pred_xgb)
# report2_xgb = classification_metrics_binary_prob(y_test, y_prob_xgb)
# report_xgb.update(report2_xgb)
# results_dict['xgb']['performance'] = report_xgb
# ```

# In[35]:


report_xgb_hist = classification_metrics_binary(y_test, y_pred_xgb_hist)
report2_xgb_hist = classification_metrics_binary_prob(y_test, y_prob_xgb_hist)
report_xgb_hist.update(report2_xgb_hist)


# In[36]:


results_dict['xgb_hist']['performance'] = report_xgb_hist


# In[37]:


report_lgbm = classification_metrics_binary(y_test, y_pred_lgbm)
report2_lgbm = classification_metrics_binary_prob(y_test, y_prob_lgbm)
report_lgbm.update(report2_lgbm)


# In[38]:


results_dict['lgbm']['performance'] = report_lgbm


# In[39]:


# Results
print(json.dumps(results_dict, indent=4, sort_keys=True))


# The experiment shows a similar performance in XGBoost hist and LightGBM. Under the parameters we used and this big dataset, LightGBM is faster. We couldn't compute the standard version of XGBoost because we got an out of memory.

# # 2) Data size benchmark
# Now we are going to analyze the performance of the libraries with different data sizes. The depth-wise implementation needs much more memory than the leaf-wise implementation.

# In[40]:


sizes = [1e4, 1e5, 1e6, 1e7]
num_rounds = 500


# In[41]:


def generate_partial_datasets(df, num_rows, test_size=0.2):
    df_small = df.sample(n=int(num_rows)).reset_index(drop=True)
    train, _, test = split_train_val_test_df(df_small, val_size=0, test_size=test_size)
    X_train, y_train = generate_feables(train)
    X_test, y_test = generate_feables(test)
    return X_train, y_train, X_test, y_test


# In[42]:


del X_train, y_train, X_test, y_test


# Let's loop for the different data sizes.

# In[43]:


for s in sizes:
    X_train, y_train, X_test, y_test = generate_partial_datasets(df_plane_numeric, s)
    clf_xgb, train_time_xgb = train_xgboost(xgb_params, X_train, y_train, num_rounds)
    y_pred, test_time_xgb = test_xgboost(clf_xgb, X_test, y_test)
    auc_xgb = roc_auc_score(y_test, y_pred)
    del clf_xgb #free GPU memory
    print("Computed XGBoost with {:.0e} samples in {:.3f}s with AUC={:.3f}".format(s, train_time_xgb, auc_xgb))
    
    clf_xgb_hist, train_time_xgb_hist = train_xgboost(xgb_hist_params, X_train, y_train, num_rounds)
    y_pred, test_time_xgb = test_xgboost(clf_xgb_hist, X_test, y_test)
    auc_xgb_hist = roc_auc_score(y_test, y_pred)
    del clf_xgb_hist
    print("Computed XGBoost hist with {:.0e} samples in {:.3f}s with AUC={:.3f}".format(s, train_time_xgb_hist, auc_xgb_hist))

    clf_lgbm, train_time_lgbm = train_lightgbm(lgbm_params, X_train, y_train, num_rounds)
    y_pred, test_time_lgbm = test_lightgbm(clf_lgbm, X_test)
    auc_lgbm = roc_auc_score(y_test, y_pred)
    del clf_lgbm
    print("Computed LightGBM with {:.0e} samples in {:.3f}s with AUC={:.3f}\n".format(s, train_time_lgbm, auc_lgbm))

