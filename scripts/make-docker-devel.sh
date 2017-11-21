#!/bin/bash
echo "Docker devel - BEGIN"
nvidia-docker build  -t opsh2oai/h2o4gpu-${versionTag}${extratag}-build -f Dockerfile-build --rm=false --build-arg cuda=${dockerimage} .
#-u `id -u`:`id -g`  -w `pwd` -v `pwd`:`pwd`:rw
nvidia-docker run --init --rm --name ${CONTAINER_NAME} -d -t -u root -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -v `pwd`:/dot  --entrypoint=bash opsh2oai/h2o4gpu-${versionTag}${extratag}-build
echo "Docker devel - Copying files"
nvidia-docker exec ${CONTAINER_NAME} bash -c 'mkdir -p repo ; cp -a /dot/. ./repo'
echo "setup pyenv, shallow clone, and make fullinstalljenkins"
nvidia-docker exec ${CONTAINER_NAME} bash -c 'eval "$(/root/.pyenv/bin/pyenv init -)" ; /root/.pyenv/bin/pyenv global 3.6.1; cd repo ; ./scripts/gitshallow_submodules.sh ; make ${makeopts} fullinstalljenkins'${extratag}' '${H2O4GPU_BUILD}' '${H2O4GPU_SUFFIX}
echo "Docker devel - Copying wheel"
nvidia-docker cp -a ${CONTAINER_NAME}:repo/src/interface_py/${dist} src/interface_py/
echo "Docker devel - Copying VERSION.txt"
mkdir -p build ; nvidia-docker cp ${CONTAINER_NAME}:repo/build/VERSION.txt build/
echo "Docker devel - Stopping docker"
nvidia-docker stop ${CONTAINER_NAME}
echo "Docker devel - END"
