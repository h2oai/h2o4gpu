# coding: utf-8
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
# # Experiment 05: Credit card Fraud (GPU version)
# 
# This experiment uses the data from the Kaggle dataset [Credit Card Fraud Detection](https://www.kaggle.com/dalpozz/creditcardfraud). The dataset is made up of a number of variables which are a result of PCA transformation.
# 
# The details of the machine we used and the version of the libraries can be found in [experiment 01](01_airline.ipynb).

# In[7]:


import json
import sys

import matplotlib.pyplot as plt
import pkg_resources
from libs.loaders import load_fraud
from libs.timer import Timer
from libs.metrics import classification_metrics_binary, classification_metrics_binary_prob, binarize_prediction
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split


print("System version: {}".format(sys.version))
print("XGBoost version: {}".format(pkg_resources.get_distribution('xgboost').version))
print("LightGBM version: {}".format(pkg_resources.get_distribution('lightgbm').version))


# In[2]:


random_seed = 42


# In[3]:


df = load_fraud()


# In[4]:


print(df.shape)
df.head()


# In[5]:


X = df[[col for col in df.columns if col.startswith('V')]].values
y = df['Class'].values


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=random_seed, test_size=0.3)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[10]:


dtrain = xgb.DMatrix(data=X_train, label=y_train, nthread=-1)
dtest = xgb.DMatrix(data=X_test, label=y_test, nthread=-1)


# In[11]:


lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train, free_raw_data=False)


# ### XGBoost

# In[72]:


results_dict = dict()
num_rounds = 100


# In[73]:


params = {'max_depth':3, 
          'objective':'binary:logistic', 
          'min_child_weight':1, 
          'eta':0.1, 
          'colsample_bytree':1, 
          'scale_pos_weight':2, 
          'gamma':0.1, 
          'reg_lamda':1, 
          'subsample':1,
          'tree_method':'gpu_exact'
          }


# In[74]:


with Timer() as t_train:
    xgb_clf_pipeline = xgb.train(params, dtrain, num_boost_round=num_rounds)
    
with Timer() as t_test:
    y_prob_xgb = xgb_clf_pipeline.predict(dtest)


# In[75]:


y_pred_xgb = binarize_prediction(y_prob_xgb)


# In[76]:


report_xgb = classification_metrics_binary(y_test, y_pred_xgb)
report2_xgb = classification_metrics_binary_prob(y_test, y_prob_xgb)
report_xgb.update(report2_xgb)


# In[77]:


results_dict['xgb']={
    'train_time': t_train.interval,
    'test_time': t_test.interval,
    'performance': report_xgb 
}


# In[78]:


del xgb_clf_pipeline


# Now let's try with XGBoost histogram.

# In[79]:


params = {'max_depth':3, 
          'objective':'binary:logistic', 
          'min_child_weight':1, 
          'eta':0.1, 
          'colsample_bytree':0.80, 
          'scale_pos_weight':2, 
          'gamma':0.1, 
          'reg_lamda':1, 
          'subsample':1,
          'tree_method':'gpu_hist'
         }


# In[80]:


with Timer() as t_train:
    xgb_hist_clf_pipeline = xgb.train(params, dtrain, num_boost_round=num_rounds)
    
with Timer() as t_test:
    y_prob_xgb_hist = xgb_hist_clf_pipeline.predict(dtest)


# In[81]:


y_pred_xgb_hist = binarize_prediction(y_prob_xgb_hist)


# In[82]:


report_xgb_hist = classification_metrics_binary(y_test, y_pred_xgb_hist)
report2_xgb_hist = classification_metrics_binary_prob(y_test, y_prob_xgb_hist)
report_xgb_hist.update(report2_xgb_hist)


# In[83]:


results_dict['xgb_hist']={
    'train_time': t_train.interval,
    'test_time': t_test.interval,
    'performance': report_xgb_hist
}


# In[84]:


del xgb_hist_clf_pipeline


# ### LightGBM

# In[85]:


params = {'num_leaves': 2**3,
         'learning_rate': 0.1,
         'scale_pos_weight': 2,
         'min_split_gain': 0.1,
         'min_child_weight': 1,
         'reg_lambda': 1,
         'subsample': 1,
         'objective':'binary',
         'task': 'train'
         }


# In[86]:


with Timer() as t_train:
    lgbm_clf_pipeline = lgb.train(params, lgb_train, num_boost_round=num_rounds)
    
with Timer() as t_test:
    y_prob_lgbm = lgbm_clf_pipeline.predict(X_test)


# In[87]:


y_pred_lgbm = binarize_prediction(y_prob_lgbm)


# In[88]:


report_lgbm = classification_metrics_binary(y_test, y_pred_lgbm)
report2_lgbm = classification_metrics_binary_prob(y_test, y_prob_lgbm)
report_lgbm.update(report2_lgbm)


# In[89]:


results_dict['lgbm']={
    'train_time': t_train.interval,
    'test_time': t_test.interval,
    'performance': report_lgbm 
}


# In[90]:


del lgbm_clf_pipeline


# Finally, we show the results

# In[91]:


# Results
print(json.dumps(results_dict, indent=4, sort_keys=True))

