
# get path
MYPWD=`pwd`
echo "PWD is $MYPWD"
export MOUNT_POINT=$MYPWD/data
echo "MOUNT_POINT is $MOUNT_POINT"
export RESULTS_DIR=$MYPWD/results
mkdir -p $RESULTS_DIR

# -1 all tests, or choose tests 0-...
runtests=-1

if [ $runtests -eq 0 ] || [ $runtests -eq -1 ]
then

# run football
cd $MOUNT_POINT/football/
unzip -o soccer.zip
cd ../../
cd tests/python/xgboost # for libs stuff
ipython 03_football_GPU.py &> $RESULTS_DIR/football.txt # py from export of ipynb removing inline commands
cd $MYPWD

fi

if [ $runtests -eq 1 ] || [ $runtests -eq -1 ]
then

# run credit
cd tests/python/xgboost # for libs stuff
ipython 05_FraudDetection_GPU.py &> $RESULTS_DIR/credit.txt # py from export of ipynb removing inline commands
cd $MYPWD

fi

if [ $runtests -eq 2 ] || [ $runtests -eq -1 ]
then

# run airlines
cd tests/python/xgboost # for libs stuff
ipython 01_airline_GPU.py &> $RESULTS_DIR/airlines.txt # py from export of ipynb removing inline commands
cd $MYPWD

fi

if [ $runtests -eq 3 ] || [ $runtests -eq -1 ]
then

# run Planet
cd $MOUNT_POINT/planet/
7z x -y train-jpg.tar.7z
tar xvf train-jpg.tar
7z x -y test-jpg.tar.7z
tar xvf test-jpg.tar
# get rid of train if no test and visa versa
numlist=`ls train-jpg test-jpg|sed 's/train_//g' | sed 's/test_//g' | sed 's/test-jpg://g'| sed 's/train-jpg://g' | sed 's/\.jpg//g' | sort|uniq -u`
#for fil in $numlist ; do echo $fil ; rm -rf test-jpg/test_$fil.jpg ; rm -rf train-jpg/train_$fil.jpg ; done
alias cp='cp'
for fil in $numlist ; do echo $fil ; cp -a train-jpg/train_$fil.jpg test-jpg/test_$fil.jpg ; done
for fil in $numlist ; do echo $fil ; cp -a test-jpg/test_$fil.jpg train-jpg/train_$fil.jpg ; done
rm -rf validate-jpg
#mkdir -p validate-jpg
#cp -a test-jpg/*.jpg validate-jpg/
cd ../../
cd tests/python/xgboost # for libs stuff
ipython 04_PlanetKaggle_GPU.py &> $RESULTS_DIR/planet.txt # py from export of ipynb removing inline commands
cd $MYPWD

fi

if [ $runtests -eq 4 ] || [ $runtests -eq -1 ]
then

# run higgs
cd tests/python/xgboost # for libs stuff
ipython 06_HIGGS_GPU.py &> $RESULTS_DIR/higgs.txt # py from export of ipynb removing inline commands
cd $MYPWD

fi


# OLD TEST CODE:
# get jupyter going
#(jupyter notebook -y --no-browser --port=28672 &> out.txt &)
#sleep 5
#link=`cat out.txt |grep http|grep token|tail -1|awk '{print $1}'`
#notelink=`echo $link| sed 's/?token.*//g'`
#fulllink=${notelink}notebooks/fast_retraining/experiments
#echo $fulllink
#chromium-browser $link &
#chromium-browser "$fulllink/03_football_GPU.ipynb"
#cd $MOUNT_POINT/football/ && unzip -o soccer.zip && cd ../../ && MOUNT_POINT=$MOUNT_POINT jupyter nbconvert --to notebook --execute fast_retraining/experiments/03_football_GPU.ipynb
#cd $(MOUNT_POINT)/football/ && unzip -o soccer.zip && cd ../../ && MOUNT_POINT=$(MOUNT_POINT) jupyter notebook  fast_retraining/experiments/03_football_GPU.ipynb                                    

