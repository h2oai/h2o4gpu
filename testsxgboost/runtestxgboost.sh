
# get path
MYPWD=`pwd`
echo "PWD is $MYPWD"
MOUNT_POINT=$MYPWD/data
echo "MOUNT_POINT is $MOUNT_POINT"
mkdir -p testsxgboost/results/

# run football
cd $MOUNT_POINT/football/
unzip -o soccer.zip
cd ../../
cd testsxgboost # for libs stuff
ipython 03_football_GPU.py &> results/football.txt # py from export of ipynb removing inline commands
cd $MYPWD

# run credit
cd testsxgboost # for libs stuff
ipython 05_FraudDetection_GPU.py &> results/credit.txt # py from export of ipynb removing inline commands
cd $MYPWD

# run airlines
cd testsxgboost # for libs stuff
ipython 01_airline_GPU.py &> results/airlines.txt # py from export of ipynb removing inline commands
cd $MYPWD

# run Planet
cd $MOUNT_POINT/planet/
7z x -y train-jpg.tar.7z
tar xvf train-jpg.tar
7z x -y test-jpg.tar.7z
tar xvf test-jpg.tar
cd ../../
cd testsxgboost # for libs stuff
ipython 04_PlanetKaggle_GPU.py &> results/planet.txt # py from export of ipynb removing inline commands
cd $MYPWD

# run higgs
cd testsxgboost # for libs stuff
ipython 06_HIGGS_GPU.py &> results/higgs.txt # py from export of ipynb removing inline commands
cd $MYPWD



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

