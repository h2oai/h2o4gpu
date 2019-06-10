#!/bin/bash
# Requires one has already done(e.g.): make docker-build-nccl-cuda9 to get wheel built or wheel was unstashed on jenkins
set -e

DOCKER_CLI='nvidia-docker'

H2O4GPU_BUILD="${H2O4GPU_BUILD:-0}"
DATA_DIRS="${DATA_DIRS:-}"

echo "Docker devel test and pylint - BEGIN"
# --build-arg http_proxy=http://172.16.2.142:3128/
$DOCKER_CLI build  -t opsh2oai/h2o4gpu-buildversion${extratag}-build -f Dockerfile-runtime --rm=false --build-arg docker_name=${dockerimage} .

#-u `id -u`:`id -g`  -w `pwd` -v `pwd`:`pwd`:rw
$DOCKER_CLI run --init --rm --name ${CONTAINER_NAME} -d -t -u root ${DATA_DIRS} -v `pwd`:/dot  --entrypoint=bash opsh2oai/h2o4gpu-buildversion${extratag}-build

echo "Docker devel test and pylint - Copying files"
$DOCKER_CLI exec ${CONTAINER_NAME} bash -c 'mkdir -p repo ; cp -a /dot/. ./repo'
$DOCKER_CLI exec ${CONTAINER_NAME} bash -c 'cd ./repo ; ln -sf /data . || true ; ln -sf /open_data . || true'

echo "Docker devel test and pylint - pip install wheel from dist/${platform}, make ${target}"

# Don't use version in wheel name when find so local call to this script works without specific jenkins versions
# Just ensure clean dist/${platform}/*.whl before unstash in jenkins
$DOCKER_CLI exec ${CONTAINER_NAME} bash -c 'export HOME=`pwd` ; cd repo ; pip install `find /dot/src/interface_py/dist/'${platform}' -name "*h2o4gpu-*.whl"`; pip freeze ; make '${target}

{ # try
    echo "Docker devel test and pylint - copy any dat results"
    rm -rf results ; mkdir -p results/
    touch results/emptyresults.dat
    $DOCKER_CLI cp -a ${CONTAINER_NAME}:repo/results results/
} || { # catch
   echo "No results dat files"
}

echo "Docker devel test and pylint - copy build reports"
rm -rf build/test-reports ; mkdir -p build/test-reports/
$DOCKER_CLI cp -a ${CONTAINER_NAME}:repo/build/test-reports build/

echo "Docker devel test and pylint - copy logs for arch"
rm -rf tmp ; mkdir -p tmp
$DOCKER_CLI cp -a ${CONTAINER_NAME}:repo/tmp ./

echo "Docker devel test and pylint - pylint"
$DOCKER_CLI exec ${CONTAINER_NAME} touch ./repo/src/interface_py/h2o4gpu/__init__.py
$DOCKER_CLI exec ${CONTAINER_NAME} bash -c 'cd repo ; make pylint'

echo "Docker devel test and pylint - stop"
$DOCKER_CLI stop ${CONTAINER_NAME}
