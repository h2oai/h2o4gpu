# coding: utf-8
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
# # Experiment 3: Football match prediction (GPU version)
# 
# In this experiment we are going to use the [Kaggle football dataset](https://www.kaggle.com/hugomathien/soccer). The dataset has information from +25,000 matches, +10,000 players from 11 European Countries with their lead championship during seasons 2008 to 2016. It also contains players attributes sourced from EA Sports' FIFA video game series. The problem we address is to try to predict if a match is going to end as win, draw or defeat. 
# 
# Part of the code use in this notebook is this [kaggle kernel](https://www.kaggle.com/airback/match-outcome-prediction-in-football).
# 
# The details of the machine we used and the version of the libraries can be found in [experiment 01](01_airline.ipynb).

# In[1]:


import os,sys
import pandas as pd
import numpy as np
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from libs.loaders import load_football
from libs.football import get_fifa_data, create_feables
from libs.timer import Timer
from libs.conversion import convert_cols_categorical_to_numeric
from libs.metrics import classification_metrics_multilabel
import pickle
import pkg_resources
import json


print("System version: {}".format(sys.version))
print("XGBoost version: {}".format(pkg_resources.get_distribution('xgboost').version))
print("LightGBM version: {}".format(pkg_resources.get_distribution('lightgbm').version))

#get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# ### Data loading and management
# 

# In[2]:


get_ipython().run_cell_magic('time', '', 'countries, matches, leagues, teams, players = load_football()\nprint(countries.shape)\nprint(matches.shape)\nprint(leagues.shape)\nprint(teams.shape)\nprint(players.shape)')


# In[3]:


leagues


# In[4]:


matches.head()


# In[5]:


#Reduce match data to fulfill run time requirements
cols = ["country_id", "league_id", "season", "stage", "date", "match_api_id", "home_team_api_id", 
        "away_team_api_id", "home_team_goal", "away_team_goal", "home_player_1", "home_player_2",
        "home_player_3", "home_player_4", "home_player_5", "home_player_6", "home_player_7", 
        "home_player_8", "home_player_9", "home_player_10", "home_player_11", "away_player_1",
        "away_player_2", "away_player_3", "away_player_4", "away_player_5", "away_player_6",
        "away_player_7", "away_player_8", "away_player_9", "away_player_10", "away_player_11"]
match_data = matches.dropna(subset = cols)
print(match_data.shape)


# Now, using the information from the matches and players, we are going to create features based on the FIFA attributes. This computation is heavy, so we are going to save it the first time we create it.  

# In[6]:


get_ipython().run_cell_magic('time', '', "fifa_data_filename = 'fifa_data.pk'\nif os.path.isfile(fifa_data_filename):\n    fifa_data = pd.read_pickle(fifa_data_filename)\nelse:\n    fifa_data = get_fifa_data(match_data, players)\n    fifa_data.to_pickle(fifa_data_filename)\nprint(fifa_data.shape)")


# Finally, we are going to compute the features and labels. The labels are related to the result of the team playing at home, they are: `Win`, `Draw`, `Defeat`. 

# In[7]:


get_ipython().run_cell_magic('time', '', "bk_cols = ['B365', 'BW', 'IW', 'LB', 'PS', 'WH', 'SJ', 'VC', 'GB', 'BS']\nbk_cols_selected = ['B365', 'BW']      \nfeables = create_feables(match_data, fifa_data, bk_cols_selected, get_overall = True)\nprint(feables.shape)")


# In[8]:


feables = convert_cols_categorical_to_numeric(feables)
feables.head()


# Let's now split features and labels.

# In[9]:


features = feables[feables.columns.difference(['match_api_id', 'label'])]
labs = feables['label']
print(features.shape)
print(labs.shape)


# Once we have the features and labels defined, let's create the train and test set.

# In[10]:


get_ipython().run_cell_magic('time', '', 'X_train, X_test, y_train, y_test = train_test_split(features, labs, test_size=0.2, random_state=42, stratify=labs)')


# In[11]:


dtrain = xgb.DMatrix(data=X_train, label=y_train, nthread=-1)
dtest = xgb.DMatrix(data=X_test, label=y_test, nthread=-1)


# In[12]:


lgb_train = lgb.Dataset(X_train.values, y_train.values, free_raw_data=False)
lgb_test = lgb.Dataset(X_test.values, y_test, reference=lgb_train, free_raw_data=False)


# ### XGBoost analysis
# Once we have done the feature engineering step, we can start to train with each of the libraries. We will start with XGBoost. 
# 
# We are going to save the training and test time, as well as some metrics. 

# In[13]:


results_dict = dict()
num_rounds = 300
labels = [0,1,2]


# In[14]:


params = {'max_depth':3, 
          'objective': 'multi:softprob', 
          'num_class': len(labels),
          'min_child_weight':5, 
          'learning_rate':0.1, 
          'colsample_bytree':0.8, 
          'scale_pos_weight':2, 
          'gamma':0.1, 
          'reg_lamda':1, 
          'subsample':1,
          'tree_method':'gpu_exact'
          }


# In[15]:


with Timer() as t_train:
    xgb_clf_pipeline = xgb.train(params, dtrain, num_boost_round=num_rounds)
    
with Timer() as t_test:
    y_prob_xgb = xgb_clf_pipeline.predict(dtest)


# In[16]:


def quantitize_multilable_prediction(y_pred):
    return np.argmax(y_pred, axis=1)


# In[17]:


y_pred_xgb = quantitize_multilable_prediction(y_prob_xgb)


# In[18]:


report_xgb = classification_metrics_multilabel(y_test, y_pred_xgb, labels)


# In[19]:


results_dict['xgb']={
    'train_time': t_train.interval,
    'test_time': t_test.interval,
    'performance': report_xgb 
}


# In[20]:


del xgb_clf_pipeline


# 
# Now let's try with XGBoost histogram.

# In[21]:


params = {'max_depth':3, 
          'objective': 'multi:softprob', 
          'num_class': len(labels),
          'min_child_weight':5, 
          'learning_rate':0.1, 
          'colsample_bytree':0.80, 
          'scale_pos_weight':2, 
          'gamma':0.1, 
          'reg_lamda':1, 
          'subsample':1,
          'tree_method':'gpu_hist'
         }


# In[22]:


with Timer() as t_train:
    xgb_hist_clf_pipeline = xgb.train(params, dtrain, num_boost_round=num_rounds)
    
with Timer() as t_test:
    y_prob_xgb_hist = xgb_hist_clf_pipeline.predict(dtest)


# In[23]:


y_pred_xgb_hist = quantitize_multilable_prediction(y_prob_xgb_hist)


# In[24]:


report_xgb_hist = classification_metrics_multilabel(y_test, y_pred_xgb_hist, labels)


# In[25]:


results_dict['xgb_hist']={
    'train_time': t_train.interval,
    'test_time': t_test.interval,
    'performance': report_xgb_hist
}


# In[26]:


del xgb_hist_clf_pipeline


# ### LightGBM analysis
# 
# Now let's compare with LightGBM.

# In[27]:


params = {'num_leaves': 2**3,
         'learning_rate': 0.1,
         'colsample_bytree': 0.80,
         'scale_pos_weight': 2,
         'min_split_gain': 0.1,
         'min_child_weight': 5,
         'reg_lambda': 1,
         'subsample': 1,
         'objective':'multiclass',
         'num_class': len(labels),
         'task': 'train'
         }


# In[28]:


with Timer() as t_train:
    lgbm_clf_pipeline = lgb.train(params, lgb_train, num_boost_round=num_rounds)
    
with Timer() as t_test:
    y_prob_lgbm = lgbm_clf_pipeline.predict(X_test.values)


# In[29]:


y_pred_lgbm = quantitize_multilable_prediction(y_prob_lgbm)


# In[30]:


report_lgbm = classification_metrics_multilabel(y_test, y_pred_lgbm, labels)


# In[31]:


results_dict['lgbm']={
    'train_time': t_train.interval,
    'test_time': t_test.interval,
    'performance': report_lgbm 
}


# In[32]:


del lgbm_clf_pipeline


# Finally, the results.

# In[33]:


# Results
print(json.dumps(results_dict, indent=4, sort_keys=True))


# As it can be seen, in the case of multilabel LightGBM is faster than XGBoost in both versions. The performance metrics are really poor, so we wouldn't recommend to bet based on this algorithm :-)

# In[ ]:




