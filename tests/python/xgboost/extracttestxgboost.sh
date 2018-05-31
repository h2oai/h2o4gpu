# get path
MYPWD=`pwd`
echo "PWD is $MYPWD"
export RESULTS_DIR=$MYPWD/results

# collect only required data
grep -B 2 -A 9 performance $RESULTS_DIR/football.txt > $RESULTS_DIR/football_acc_perf.json
grep -B 2 -A 10 performance $RESULTS_DIR/credit.txt > $RESULTS_DIR/credit_acc_perf.json     # also has AUC
grep -B 2 -A 8 performance $RESULTS_DIR/airlines.txt > $RESULTS_DIR/airlines_acc_perf.json
grep -B 2 -A 8 performance $RESULTS_DIR/planet.txt > $RESULTS_DIR/planet_acc_perf.json
grep -B 2 -A 8 performance $RESULTS_DIR/higgs.txt > $RESULTS_DIR/higgs_acc_perf.json

# extract results out of the json
python tests/python/xgboost/extractjson.py test_gbm_football $RESULTS_DIR $RESULTS_DIR/football_acc_perf.json $RESULTS_DIR/test_gbm_football.error.dat $RESULTS_DIR/test_gbm_football.error.h2o.dat $RESULTS_DIR/test_gbm_football.error.h2o4gpu.dat $RESULTS_DIR/test_gbm_football.time.dat $RESULTS_DIR/test_gbm_football.time.h2o.dat $RESULTS_DIR/test_gbm_football.time.h2o4gpu.dat
python tests/python/xgboost/extractjson.py test_gbm_credit $RESULTS_DIR $RESULTS_DIR/credit_acc_perf.json $RESULTS_DIR/test_gbm_credit.error.dat $RESULTS_DIR/test_gbm_credit.error.h2o.dat $RESULTS_DIR/test_gbm_credit.error.h2o4gpu.dat $RESULTS_DIR/test_gbm_credit.time.dat $RESULTS_DIR/test_gbm_credit.time.h2o.dat $RESULTS_DIR/test_gbm_credit.time.h2o4gpu.dat
python tests/python/xgboost/extractjson.py test_gbm_airlines $RESULTS_DIR $RESULTS_DIR/airlines_acc_perf.json $RESULTS_DIR/test_gbm_airlines.error.dat $RESULTS_DIR/test_gbm_airlines.error.h2o.dat $RESULTS_DIR/test_gbm_airlines.error.h2o4gpu.dat $RESULTS_DIR/test_gbm_airlines.time.dat $RESULTS_DIR/test_gbm_airlines.time.h2o.dat $RESULTS_DIR/test_gbm_airlines.time.h2o4gpu.dat
python tests/python/xgboost/extractjson.py test_gbm_planet $RESULTS_DIR $RESULTS_DIR/planet_acc_perf.json $RESULTS_DIR/test_gbm_planet.error.dat $RESULTS_DIR/test_gbm_planet.error.h2o.dat $RESULTS_DIR/test_gbm_planet.error.h2o4gpu.dat $RESULTS_DIR/test_gbm_planet.time.dat $RESULTS_DIR/test_gbm_planet.time.h2o.dat $RESULTS_DIR/test_gbm_planet.time.h2o4gpu.dat
python tests/python/xgboost/extractjson.py test_gbm_higgs $RESULTS_DIR $RESULTS_DIR/higgs_acc_perf.json $RESULTS_DIR/test_gbm_higgs.error.dat $RESULTS_DIR/test_gbm_higgs.error.h2o.dat $RESULTS_DIR/test_gbm_higgs.error.h2o4gpu.dat $RESULTS_DIR/test_gbm_higgs.time.dat $RESULTS_DIR/test_gbm_higgs.time.h2o.dat $RESULTS_DIR/test_gbm_higgs.time.h2o4gpu.dat

