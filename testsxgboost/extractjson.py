"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import json
import sys, os

args = sys.argv

name = args[1]
path = args[2]
inputfilename = args[3]
print("arg3=%s" % inputfilename)
# same file pointer conventions as in glm and kmeans tests
# error
fn1 = args[4]
fn1a = args[5]
fn1b = args[6]
# time
fn2 = args[7]
fn2a = args[8]
fn2b = args[9]

data = json.load(open(inputfilename))

acc_lgb =  data["lgbm"]["performance"]["Accuracy"]
acc_xgb =  data["xgb_hist"]["performance"]["Accuracy"]

time_lgb =  data["lgbm"]["train_time"] + data["lgbm"]["test_time"]
time_xgb =  data["xgb_hist"]["train_time"] + data["xgb_hist"]["test_time"]

os.makedirs(path, exist_ok=True)

if True:
    # "ERROR" = 1/acc
    print("ERROR")
    
    f1 = open(fn1, 'wt+')
    f1a = open(fn1a, 'wt+')
    f1b = open(fn1b, 'wt+')
    
    thisrelerror = (acc_xgb - acc_lgb)/(abs(acc_xgb) + abs(acc_lgb))
    h2o_train_error = 1.0/acc_lgb
    error_train = 1.0/acc_xgb
    
    print('%s' % (name), file=f1, end="")
    if thisrelerror>0:
        print(' OK', file=f1)
    else:
        print(' %g' % thisrelerror, file=f1)
    print('%s' % (name), file=f1a, end="")
    print(' %g' % h2o_train_error, file=f1a)
    print('%s' % (name), file=f1b, end="")
    print(' %g' % error_train, file=f1b)
    
if True:
    # TIME
    print("TIME")
    
    f2 = open(fn2, 'wt+')
    f2a = open(fn2a, 'wt+')
    f2b = open(fn2b, 'wt+')
    
    duration_h2o = time_lgb
    duration_h2o4gpu = time_xgb
    ratio_time = duration_h2o4gpu/duration_h2o
    
    print('%s' % (name), file=f2, end="")
    print(' %g' % ratio_time, file=f2)
    print('%s' % (name), file=f2a, end="")
    print(' %g' % duration_h2o, file=f2a)
    print('%s' % (name), file=f2b, end="")
    print(' %g' % duration_h2o4gpu, file=f2b)
    
