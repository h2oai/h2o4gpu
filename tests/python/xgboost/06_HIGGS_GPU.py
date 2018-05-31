# coding: utf-8
"""
:copyright: 2017-2018 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
# # Experiment 06: HIGGS boson (GPU version)
# 
# This experiment uses the data from the [HIGGS dataset](https://archive.ics.uci.edu/ml/datasets/HIGGS) to predict the appearance of the Higgs boson. The dataset consists of 11 million of observations. More information about the data can be found in [loaders.py](libs/loaders.py).  
# 
# The details of the machine we used and the version of the libraries can be found in [experiment 01](01_airline.ipynb).

# In[25]:


import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import json
import seaborn
import matplotlib.pyplot as plt
import pkg_resources
from libs.loaders import load_higgs
from libs.timer import Timer
from libs.metrics import classification_metrics_binary, classification_metrics_binary_prob, binarize_prediction
import warnings

print("System version: {}".format(sys.version))
print("XGBoost version: {}".format(pkg_resources.get_distribution('xgboost').version))
print("LightGBM version: {}".format(pkg_resources.get_distribution('lightgbm').version))

warnings.filterwarnings("ignore")
#get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# ### Data loading and management

# In[2]:


get_ipython().run_cell_magic('time', '', 'df = load_higgs()\nprint(df.shape)')


# In[3]:


df.head(5)


# Depending on your GPU you could experiment memory issues, if that is so, you could try to reduce the datasize. 

# In[4]:


#subset = 1e6
#df_small = df.sample(n=subset).reset_index(drop=True)


# Let's generate the train and test set.

# In[5]:


def generate_feables(df):
    X = df[df.columns.difference(['boson'])]
    y = df['boson']
    return X,y


# In[6]:


get_ipython().run_cell_magic('time', '', 'X, y = generate_feables(df)\n#X, y = generate_feables(df_small)\nprint(X.shape)\nprint(y.shape)')


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=77, test_size=500000)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# Let's put the data in the XGBoost format.

# In[8]:


dtrain = xgb.DMatrix(data=X_train, label=y_train, nthread=-1)
dtest = xgb.DMatrix(data=X_test, label=y_test, nthread=-1)


# Now, we'll do the same for LightGBM.

# In[9]:


lgb_train = lgb.Dataset(X_train.values, y_train.values, free_raw_data=False)
lgb_test = lgb.Dataset(X_test.values, y_test.values, reference=lgb_train, free_raw_data=False)


# ### XGBoost
# Let's start by computing the standard version of XGBoost in a GPU.

# In[10]:


results_dict = dict()
num_rounds = 200


# In[11]:


params = {'max_depth':2, #'max_depth':5, 
          'objective':'binary:logistic', 
          'min_child_weight':1, 
          'learning_rate':0.1, 
          'scale_pos_weight':2, 
          'gamma':0.1, 
          'reg_lamda':1, 
          'subsample':1,
          'tree_method':'gpu_exact'
          }


# *NOTE: We got an out of memory error with xgb. Please see the comments at the end of the notebook.*

# ```python
# with Timer() as train_t:
#     xgb_clf_pipeline = xgb.train(params, dtrain, num_boost_round=num_rounds)
#     
# with Timer() as test_t:
#     y_prob_xgb = xgb_clf_pipeline.predict(dtest)
#     
# ```

# Once the training and test is finised, let's compute some metrics.

# ```python
# y_pred_xgb = binarize_prediction(y_prob_xgb)
# report_xgb = classification_metrics_binary(y_test, y_pred_xgb)
# report2_xgb = classification_metrics_binary_prob(y_test, y_prob_xgb)
# report_xgb.update(report2_xgb)
# results_dict['xgb']={
#     'train_time': train_t.interval,
#     'test_time': test_t.interval,
#     'performance': report_xgb 
# }
# del xgb_clf_pipeline 
# 
# ```

# Now let's try with XGBoost histogram.

# In[12]:


params = {'max_depth':2, 
          'objective':'binary:logistic', 
          'min_child_weight':1, 
          'learning_rate':0.1, 
          'scale_pos_weight':2, 
          'gamma':0.1, 
          'reg_lamda':1, 
          'subsample':1,
          'tree_method':'gpu_hist'
         }


# In[13]:


with Timer() as t_train:
    xgb_hist_clf_pipeline = xgb.train(params, dtrain, num_boost_round=num_rounds)
    
with Timer() as t_test:
    y_prob_xgb_hist = xgb_hist_clf_pipeline.predict(dtest)


# In[14]:


y_pred_xgb_hist = binarize_prediction(y_prob_xgb_hist)


# In[15]:


report_xgb_hist = classification_metrics_binary(y_test, y_pred_xgb_hist)
report2_xgb_hist = classification_metrics_binary_prob(y_test, y_prob_xgb_hist)
report_xgb_hist.update(report2_xgb_hist)


# In[16]:


results_dict['xgb_hist']={
    'train_time': t_train.interval,
    'test_time': t_test.interval,
    'performance': report_xgb_hist
}


# In[17]:


del xgb_hist_clf_pipeline #clear GPU memory (214Mb)


# ### LightGBM
# After the XGBoost version is finished, let's try LightGBM in GPU. 

# In[18]:


params = {'num_leaves': 2**5,
         'learning_rate': 0.1,
         'scale_pos_weight': 2,
         'min_split_gain': 0.1,
         'min_child_weight': 1,
         'reg_lambda': 1,
         'subsample': 1,
         'objective':'binary',
         'device': 'gpu',
         'task': 'train'
         }


# In[19]:


with Timer() as train_t:
    lgbm_clf_pipeline = lgb.train(params, lgb_train, num_boost_round=num_rounds)
    
with Timer() as test_t:
    y_prob_lgbm = lgbm_clf_pipeline.predict(X_test.values)


# As we did before, let's obtain some performance metrics.

# In[20]:


y_pred_lgbm = binarize_prediction(y_prob_lgbm)


# In[21]:


report_lgbm = classification_metrics_binary(y_test, y_pred_lgbm)
report2_lgbm = classification_metrics_binary_prob(y_test, y_prob_lgbm)
report_lgbm.update(report2_lgbm)


# In[22]:


results_dict['lgbm']={
    'train_time': train_t.interval,
    'test_time': test_t.interval,
    'performance': report_lgbm 
}


# In[23]:


del lgbm_clf_pipeline #clear GPU memory (135Mb)


# Finally, we show the results

# In[24]:


# Results
print(json.dumps(results_dict, indent=4, sort_keys=True))


# The full size of HIGGS dataset is 11 million rows. This amount of information can not be processed by XGBoost in its standard version (xgb) using a NVIDIA M60 GPU, even if we reduce the max depth of the tree to 2. We got an out of memory error. However, when reducing the dataset to 1 million rows, xgb works correctly. 
# 
# In our experiments with the reduced dataset of 1 million rows, the memory consumption of xgb is around 10 times higher than LightGBM and 5 times higher than XGBoost histogram (leaf-wise implementation).
# 
# We can observe that LightGBM is faster than XGBoost histogram, having a similar performance. But also, when we did the experiment with the reduced dataset, we found that  XGBoost with the leaf-wise implementation is faster than with the depth-wise implementation. 
# 
# Final advice: go leaf-wise :-)
