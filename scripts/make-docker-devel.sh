#!/bin/bash
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


#--build-arg http_proxy=http://172.16.2.142:3128/
echo "Docker devel - BEGIN"
$docker build  -t opsh2oai/h2o4gpu-buildversion${extratag}-build -f Dockerfile-build --rm=false --build-arg layer=$layer --build-arg version=$version .
#-u `id -u`:`id -g`  -w `pwd` -v `pwd`:`pwd`:rw
$docker run --init --rm --name ${CONTAINER_NAME} -d -t -u root -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -v `pwd`:/dot  --entrypoint=bash opsh2oai/h2o4gpu-buildversion${extratag}-build

echo "Docker devel - Copying files"
$docker exec ${CONTAINER_NAME} bash -c 'mkdir -p repo ; cp -a /dot/. ./repo'

echo "setup pyenv, shallow clone, and make fullinstalljenkins with ${H2O4GPU_BUILD} and ${H2O4GPU_SUFFIX}"
$docker exec ${CONTAINER_NAME} bash -c "eval \"\$(/root/.pyenv/bin/pyenv init -)\" ; /root/.pyenv/bin/pyenv global 3.6.1; cd repo ; ./scripts/gitshallow_submodules.sh ; make ${makeopts} fullinstalljenkins${extratag} H2O4GPU_BUILD=${H2O4GPU_BUILD} H2O4GPU_SUFFIX=${H2O4GPU_SUFFIX}"

echo "Docker devel - Clean local wheels and Copying wheel from docker"
rm -rf src/interface_py/${dist}/*.whl
$docker cp -a ${CONTAINER_NAME}:repo/src/interface_py/${dist} src/interface_py/

echo "Docker devel - Copying VERSION.txt"
mkdir -p build ; $docker cp ${CONTAINER_NAME}:repo/build/VERSION.txt build/

echo "Docker devel - Stopping docker"
$docker stop ${CONTAINER_NAME}

echo "Docker devel - END"
