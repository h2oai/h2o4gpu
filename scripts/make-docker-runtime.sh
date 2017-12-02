#!/bin/bash
set -e
echo "Docker runtime - BEGIN"

echo "Docker runtime - Build"
# wheel=${encodedFullVersionTag}${extratag}/h2o4gpu-${encodedFullVersionTag}-py36-none-any.whl # use this if want to pull from s3 in Dockerfile-runtime
nvidia-docker build -t opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime:latest -f Dockerfile-runtime --rm=false --build-arg cuda=${dockerimage} .
# -u `id -u`:`id -g` -d -t -w `pwd` -v `pwd`:`pwd`:rw

echo "Runtime Docker - Run"
nvidia-docker run --init --rm --name ${CONTAINER_NAME} -d -t -u root -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -v `pwd`:/dot  --entrypoint=bash opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime:latest

echo "Docker runtime - pip install h2o4gpu and pip freeze"
nvidia-docker exec ${CONTAINER_NAME} bash -c '. /h2o4gpu_env/bin/activate ; pip install `find /dot/src/interface_py/'${dist}' -name "*h2o4gpu-*.whl"` ; pip freeze'

{ # try
    echo "Docker runtime - Getting Data"
    #nvidia-docker exec ${CONTAINER_NAME} bash -c '. /h2o4gpu_env/bin/activate ; mkdir -p scripts ; rm -rf scripts/fcov_get.py ; echo "from sklearn.datasets import fetch_covtype" > ./scripts/fcov_get.py ; echo "cov = fetch_covtype()" >> ./scripts/fcov_get.py'
    #nvidia-docker exec ${CONTAINER_NAME} bash -c '. /h2o4gpu_env/bin/activate ; cd /jupyter/ ; python ../scripts/fcov_get.py'
    nvidia-docker exec ${CONTAINER_NAME} bash -c 'cd /jupyter/ ; mkdir -p ./scikit_learn_data/covertype ; cp /open_data/covertype/* ./scikit_learn_data/covertype'
    nvidia-docker exec ${CONTAINER_NAME} bash -c 'cd /jupyter/ ; mkdir -p ./scikit_learn_data/lfw_home ; cp /open_data/lfw_home/* ./scikit_learn_data/lfw_home'
    nvidia-docker exec ${CONTAINER_NAME} bash -c 'cd /jupyter/demos ; cp /data/creditcard.csv .'
    nvidia-docker exec ${CONTAINER_NAME} bash -c 'cd /jupyter/demos ; wget https://s3.amazonaws.com/h2o-public-test-data/h2o4gpu/open_data/kmeans_data/h2o-logo.jpg'
    nvidia-docker exec ${CONTAINER_NAME} bash -c 'cd /jupyter/demos ; cp /data/ipums_1k.csv .'
    nvidia-docker exec ${CONTAINER_NAME} bash -c 'cd /jupyter/demos ; cp /data/ipums.feather .'
} || { # catch
   echo "Some Data Not Obtained"
}
nvidia-docker commit ${CONTAINER_NAME} opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime:latest

echo "Docker runtime - stopping docker"
nvidia-docker stop ${CONTAINER_NAME}

if [ -z `command -v pbzip2` ]
then
    echo "Docker runtime - saving docker to local disk -- native system must have bzip2"
    nvidia-docker save opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime | bzip2 > h2o4gpu-${fullVersionTag}${extratag}-runtime.tar.bz2
else
    echo "Docker runtime - saving docker to local disk -- native system must have pbzip2"
    nvidia-docker save opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime | pbzip2 > h2o4gpu-${fullVersionTag}${extratag}-runtime.tar.bz2
fi

echo "Docker runtime - END"
