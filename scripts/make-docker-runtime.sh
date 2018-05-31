#!/bin/bash
set -e

DOCKER_CLI='nvidia-docker'

DATA_DIRS="${DATA_DIRS:-}"

echo "Docker runtime - BEGIN"

echo "Docker runtime - Build"
# wheel=${encodedFullVersionTag}${extratag}/h2o4gpu-${encodedFullVersionTag}-cp36-cp36m-linux_x86_64.whl # use this if want to pull from s3 in Dockerfile-runtime
#  --build-arg http_proxy=http://172.16.2.142:3128/
$DOCKER_CLI build -t opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime:latest -f Dockerfile-runtime --rm=false --build-arg docker_name=${dockerimage} .
# -u `id -u`:`id -g` -d -t -w `pwd` -v `pwd`:`pwd`:rw

echo "Runtime Docker - Run"
$DOCKER_CLI run --init --rm --name ${CONTAINER_NAME} -d -t -u root ${DATA_DIRS} -v `pwd`:/dot  --entrypoint=bash opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime:latest

echo "Docker runtime - pip install h2o4gpu and pip freeze"
$DOCKER_CLI exec ${CONTAINER_NAME} bash -c 'pip install `find /dot/src/interface_py/dist/'${platform}' -name "*h2o4gpu-*.whl" | xargs ls -tr | tail -1` ; pip freeze'

{ # try
    echo "Docker runtime - Getting Data"
    #$DOCKER_CLI exec ${CONTAINER_NAME} bash -c 'mkdir -p scripts ; rm -rf scripts/fcov_get.py ; echo "from sklearn.datasets import fetch_covtype" > ./scripts/fcov_get.py ; echo "cov = fetch_covtype()" >> ./scripts/fcov_get.py'
    #$DOCKER_CLI exec ${CONTAINER_NAME} bash -c 'cd /jupyter/ ; python ../scripts/fcov_get.py'
    $DOCKER_CLI exec ${CONTAINER_NAME} bash -c 'cd /jupyter/ ; mkdir -p ./scikit_learn_data/covertype ; cp /open_data/covertype/* ./scikit_learn_data/covertype'
    $DOCKER_CLI exec ${CONTAINER_NAME} bash -c 'cd /jupyter/ ; mkdir -p ./scikit_learn_data/lfw_home ; cp -af /open_data/lfw_home ./scikit_learn_data'
    $DOCKER_CLI exec ${CONTAINER_NAME} bash -c 'cd /jupyter/demos ; cp /data/creditcard.csv .'
    $DOCKER_CLI exec ${CONTAINER_NAME} bash -c 'cd /jupyter/demos ; wget https://s3.amazonaws.com/h2o-public-test-data/h2o4gpu/open_data/kmeans_data/h2o-logo.jpg'
    $DOCKER_CLI exec ${CONTAINER_NAME} bash -c 'cd /jupyter/demos ; wget https://s3.amazonaws.com/h2o-public-test-data/h2o4gpu/open_data/Temples-shrines-and-castles-in-Japan-social-media-image.jpg'
    $DOCKER_CLI exec ${CONTAINER_NAME} bash -c 'cd /jupyter/demos ; wget https://s3.amazonaws.com/h2o-public-test-data/h2o4gpu/open_data/china.jpg'
    $DOCKER_CLI exec ${CONTAINER_NAME} bash -c 'cd /jupyter/demos ; cp /data/ipums_1k.csv .'
    $DOCKER_CLI exec ${CONTAINER_NAME} bash -c 'cd /jupyter/demos ; cp /data/ipums.feather .'
    $DOCKER_CLI exec -u root ${CONTAINER_NAME} bash -c 'cd /jupyter/ ; chmod -R a+rwx .'
} || { # catch
   echo "Some Data Not Obtained"
}
$DOCKER_CLI commit ${CONTAINER_NAME} opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime:latest

echo "Docker runtime - stopping docker"
$DOCKER_CLI stop ${CONTAINER_NAME}

if [ -z `command -v pbzip2` ]
then
    echo "Docker runtime - saving docker to local disk -- native system must have bzip2"
    $DOCKER_CLI save opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime | bzip2 > h2o4gpu-${fullVersionTag}${extratag}-runtime.tar.bz2
else
    echo "Docker runtime - saving docker to local disk -- native system must have pbzip2"
    $DOCKER_CLI save opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime | pbzip2 > h2o4gpu-${fullVersionTag}${extratag}-runtime.tar.bz2
fi

echo "Docker runtime - END"
