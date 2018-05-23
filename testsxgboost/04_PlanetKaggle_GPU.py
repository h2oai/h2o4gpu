# coding: utf-8
"""
:copyright: 2017-2018 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
# # Experiment 04: Amazon Planet (GPU version)
# 
# This experiment uses the data from the Kaggle competition [Planet: Understanding the Amazon from Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/leaderboard). Here we use a pretrained ResNet50 model to generate the features from the dataset.
# 
# The details of the machine we used and the version of the libraries can be found in [experiment 01](01_airline.ipynb).

# In[1]:


import sys, os
from collections import defaultdict
import numpy as np
import pkg_resources
from libs.loaders import load_planet_kaggle
from libs.planet_kaggle import threshold_prediction
from libs.timer import Timer
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session, get_session

print("System version: {}".format(sys.version))
print("XGBoost version: {}".format(pkg_resources.get_distribution('xgboost').version))
print("LightGBM version: {}".format(pkg_resources.get_distribution('lightgbm').version))


# In[2]:


#get_ipython().magic('env MOUNT_POINT=/datadrive')


# In[3]:


#Configure TF to use only one GPU, by default TF allocates memory in all GPUs
config = tf.ConfigProto(device_count = {'GPU': 1})
#Configure TF to limit the amount of GPU memory, by default TF takes all of them. 
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))


# The images are loaded and featurised using a pretrained ResNet50 model available from Keras

# In[4]:


X_train, y_train, X_test, y_test = load_planet_kaggle()


# In[5]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# ## XGBoost 

# We will use a one-v-rest. So each classifier will be responsible for determining whether the assigned tag applies to the image

# In[6]:


def train_and_validate_xgboost(params, train_features, train_labels, validation_features, num_boost_round):
    n_classes = train_labels.shape[1]
    y_val_pred = np.zeros((validation_features.shape[0], n_classes))
    time_results = defaultdict(list)
    for class_i in tqdm(range(n_classes)):
        dtrain = xgb.DMatrix(data=train_features, label=train_labels[:, class_i], nthread=-1)
        dtest = xgb.DMatrix(data=validation_features, nthread=-1)
        with Timer() as t:
            model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
        time_results['train_time'].append(t.interval)
        
        with Timer() as t:
            y_val_pred[:, class_i] = model.predict(dtest)
        time_results['test_time'].append(t.interval)
        
    return y_val_pred, time_results


# In[7]:


def train_and_validate_lightgbm(params, train_features, train_labels, validation_features, num_boost_round):
    n_classes = train_labels.shape[1]
    y_val_pred = np.zeros((validation_features.shape[0], n_classes))
    time_results = defaultdict(list)
    for class_i in tqdm(range(n_classes)):
        lgb_train = lgb.Dataset(train_features, train_labels[:, class_i], free_raw_data=False)
        with Timer() as t:
            model = lgb.train(params, lgb_train, num_boost_round = num_boost_round)
        time_results['train_time'].append(t.interval)
        
        with Timer() as t:
            y_val_pred[:, class_i] = model.predict(validation_features)
        time_results['test_time'].append(t.interval)
        
    return y_val_pred, time_results


# In[8]:


metrics_dict = {
    'Accuracy': accuracy_score,
    'Precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='samples'),
    'Recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='samples'),
    'F1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='samples'),
}

def classification_metrics(metrics, y_true, y_pred):
    return {metric_name:metric(y_true, y_pred) for metric_name, metric in metrics.items()}


# In[9]:


results_dict = dict()
num_rounds = 50


# Now we are going to define the different models.

# In[10]:


xgb_params = {'max_depth':2, #'max_depth':6 
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

# In[ ]:


y_pred, timing_results = train_and_validate_xgboost(xgb_params, X_train, y_train, X_test, num_boost_round=num_rounds)


# In[ ]:


results_dict['xgb']={
    'train_time': np.sum(timing_results['train_time']),
    'test_time': np.sum(timing_results['test_time']),
    'performance': classification_metrics(metrics_dict, 
                                          y_test, 
                                          threshold_prediction(y_pred, threshold=0.1)) 
}


# 
# 
# Now let's try with XGBoost histogram.
# 

# In[12]:


xgb_hist_params = {'max_depth':2, 
                  'objective':'binary:logistic', 
                  'min_child_weight':1, 
                  'learning_rate':0.1, 
                  'scale_pos_weight':2, 
                  'gamma':0.1, 
                  'reg_lamda':1, 
                  'subsample':1,
                  'tree_method':'gpu_hist', 
                  'max_bins': 63
                 }


# In[ ]:


y_pred, timing_results = train_and_validate_xgboost(xgb_hist_params, X_train, y_train, X_test, num_boost_round=num_rounds)


# In[ ]:


results_dict['xgb_hist']={
    'train_time': np.sum(timing_results['train_time']),
    'test_time': np.sum(timing_results['test_time']),
    'performance': classification_metrics(metrics_dict, 
                                          y_test, 
                                          threshold_prediction(y_pred, threshold=0.1)) 
}


# ## LightGBM 
# 
# 

# In[21]:


lgb_params = {'num_leaves': 2**6,
             'learning_rate': 0.1,
             'scale_pos_weight': 2,
             'min_split_gain': 0.1,
             'min_child_weight': 1,
             'reg_lambda': 1,
             'subsample': 1,
             'objective':'binary',
             'device': 'gpu',
             'task': 'train',
             'max_bin': 63
             }


# In[22]:


y_pred, timing_results = train_and_validate_lightgbm(lgb_params, X_train, y_train, X_test, num_boost_round=num_rounds)


# In[23]:


results_dict['lgbm']={
    'train_time': np.sum(timing_results['train_time']),
    'test_time': np.sum(timing_results['test_time']),
    'performance': classification_metrics(metrics_dict, 
                                          y_test, 
                                          threshold_prediction(y_pred, threshold=0.1)) 
}


# Finally, we show the results.

# In[24]:


# Results
print(json.dumps(results_dict, indent=4, sort_keys=True))


# In this dataset we have a big feature size, 2048. When using the standard version of XGBoost, xgb, we get an out of memory using a NVIDIA M60 GPU, even if we reduce the max depth of the tree to 2. A solution to this issue would be to reduce the feature size. One option could be using PCA and another could be to use a different featurizer, instead of ResNet whose last hidden layer has 2048 units, we could use VGG, [also provided by Keras](https://github.com/fchollet/keras/blob/master/keras/applications/vgg16.py), whose last hidden layer has 512 units. 
# 
# As it can be seen, LightGBM is faster than XGBoost, but in this case the speed is lower than in the CPU version. The GPU implementation cannot always speed up the training, since it has some additional cost of memory copy between CPU and GPU. So when the data size is small and the number of features is large, the GPU version will be slower.
