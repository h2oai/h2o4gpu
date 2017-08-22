# get path
PWD=`pwd`
MOUNT_POINT=$PWD/data

# run football
cd $MOUNT_POINT/football/
unzip -o soccer.zip
cd ../../
cd fast_retraining/experiments # for libs stuff
ipython ../../testsxgboost/03_football_GPU.py &> football.txt # py from export of ipynb removing inline commands


# run credit
cd fast_retraining/experiments # for libs stuff
ipython ../../testsxgboost/05_FraudDetection_GPU.py &> credit.txt # py from export of ipynb removing inline commands

# run airlines
cd fast_retraining/experiments # for libs stuff
ipython ../../testsxgboost/01_airline_GPU.py &> airlines.txt # py from export of ipynb removing inline commands


# run Planet
cd $MOUNT_POINT/planet/
7z x train-jpg.tar.7z
tar xvf train-jpg.tar
7z x test-jpg.tar.7z
tar xvf test-jpg.tar
cd ../../
cd fast_retraining/experiments # for libs stuff
ipython ../../testsxgboost/04_PlanetKaggle_GPU.py &> planet.txt # py from export of ipynb removing inline commands

# run higgs
cd fast_retraining/experiments # for libs stuff
ipython ../../testsxgboost/06_higgs_GPU.py &> higgs.txt # py from export of ipynb removing inline commands



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

