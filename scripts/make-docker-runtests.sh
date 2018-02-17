#!/bin/bash
# Requires one has already done(e.g.): make docker-build-nccl-cuda9 to get wheel built or wheel was unstashed on jenkins
set -e

# split layer and version
IFS=':' read -ra LAYER_VERSION <<< "${dockerimage}"
layer=${LAYER_VERSION[0]}
version=${LAYER_VERSION[1]}

if [ "$layer" == "ubuntu" ]
then
	docker=docker
else
	docker=nvidia-docker
fi


echo "Docker devel test and pylint - BEGIN"
# --build-arg http_proxy=http://172.16.2.142:3128/
$docker build  -t opsh2oai/h2o4gpu-buildversion${extratag}-build -f Dockerfile-build --rm=false --build-arg layer=$layer --build-arg version=$version .
#-u `id -u`:`id -g`  -w `pwd` -v `pwd`:`pwd`:rw
$docker run --init --rm --name ${CONTAINER_NAME} -d -t -u root -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -v `pwd`:/dot  --entrypoint=bash opsh2oai/h2o4gpu-buildversion${extratag}-build

echo "Docker devel test and pylint - Copying files"
$docker exec ${CONTAINER_NAME} bash -c 'mkdir -p repo ; cp -a /dot/. ./repo'
$docker exec ${CONTAINER_NAME} bash -c 'cd ./repo ; ln -sf /data . || true ; ln -sf /open_data . || true'

echo "Docker devel test and pylint - setup pyenv, pip install wheel from ${dist}, make ${target}"

# Don't use version in wheel name when find so local call to this script works without specific jenkins versions
# Just ensure clean ${dist}/*.whl before unstash in jenkins
$docker exec ${CONTAINER_NAME} bash -c 'export HOME=`pwd`; eval "$(/root/.pyenv/bin/pyenv init -)" ; /root/.pyenv/bin/pyenv global 3.6.1; cd repo ; pip install `find /dot/src/interface_py/'${dist}' -name "*h2o4gpu-*.whl"`; pip freeze ; make '${target}

{ # try
    echo "Docker devel test and pylint - copy any dat results"
    rm -rf results ; mkdir -p results/
    touch results/emptyresults.dat
    nvidia-docker cp -a ${CONTAINER_NAME}:repo/results results/
} || { # catch
   echo "No results dat files"
}

echo "Docker devel test and pylint - copy build reports"
rm -rf build/test-reports ; mkdir -p build/test-reports/
$docker cp -a ${CONTAINER_NAME}:repo/build/test-reports build/

echo "Docker devel test and pylint - copy logs for arch"
rm -rf tmp ; mkdir -p tmp
$docker cp -a ${CONTAINER_NAME}:repo/tmp ./

echo "Docker devel test and pylint - pylint"
$docker exec ${CONTAINER_NAME} touch ./repo/src/interface_py/h2o4gpu/__init__.py
$docker exec ${CONTAINER_NAME} bash -c 'eval "$(/root/.pyenv/bin/pyenv init -)"  ;  /root/.pyenv/bin/pyenv global 3.6.1; cd repo ; make pylint'

echo "Docker devel test and pylint - stop"
$docker stop ${CONTAINER_NAME}
