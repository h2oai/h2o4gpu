#!/bin/bash
echo "Docker devel test and pylint - BEGIN"
nvidia-docker build  -t opsh2oai/h2o4gpu-${versionTag}${extratag}-build -f Dockerfile-build --rm=false --build-arg cuda=${dockerimage} .
#-u `id -u`:`id -g`  -w `pwd` -v `pwd`:`pwd`:rw
nvidia-docker run --init --rm --name ${CONTAINER_NAME} -d -t -u root -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -v `pwd`:/dot  --entrypoint=bash opsh2oai/h2o4gpu-${versionTag}${extratag}-build
echo "Docker devel test and pylint - Copying files"
nvidia-docker exec ${CONTAINER_NAME} bash -c 'mkdir -p repo ; cp -a /dot/. ./repo ; cd ./repo ; ln -sf /data . ; ln -sf /open_data .'
echo "Docker devel test and pylint - setup pyenv, pip install from ${dist} wheel with versionTag=${versionTag}, make ${target}"
nvidia-docker exec ${CONTAINER_NAME} bash -c 'export HOME=`pwd`; eval "$(/root/.pyenv/bin/pyenv init -)" ; /root/.pyenv/bin/pyenv global 3.6.1; cd repo ; pip install `find /dot/src/interface_py/'${dist}' -name "*h2o4gpu-'${versionTag}'*.whl"`; pip freeze ; make '${target}
nvidia-docker exec ${CONTAINER_NAME} touch ./repo/src/interface_py/h2o4gpu/__init__.py
echo "Docker devel test and pylint - pylint"
nvidia-docker exec ${CONTAINER_NAME} bash -c 'eval "$(/root/.pyenv/bin/pyenv init -)"  ;  /root/.pyenv/bin/pyenv global 3.6.1; cd repo ; make pylint'
echo "Docker devel test and pylint - copy build reports"
rm -rf build/test-reports ; mkdir -p build/test-reports/
nvidia-docker cp -a ${CONTAINER_NAME}:repo/build/test-reports build/
echo "Docker devel test and pylint - stop"
nvidia-docker stop ${CONTAINER_NAME}
