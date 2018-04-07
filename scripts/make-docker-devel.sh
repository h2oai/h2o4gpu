#!/bin/bash
set -e

H2O4GPU_BUILD="${H2O4GPU_BUILD:-0}"
H2O4GPU_SUFFIX="${H2O4GPU_SUFFIX:-''}"
CONTAINER_NAME="${CONTAINER_NAME:-$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)}"
makeopts="${makeopts:-}"

DOCKER_CLI='nvidia-docker'

#--build-arg http_proxy=http://172.16.2.142:3128/
echo "Docker devel - BEGIN"
$DOCKER_CLI build -t opsh2oai/h2o4gpu-buildversion${extratag}-build -f Dockerfile-build --rm=false --build-arg docker_name=${dockerimage} .

$DOCKER_CLI run --init --rm --name ${CONTAINER_NAME} -d -t -u root -v `pwd`:/dot  --entrypoint=bash opsh2oai/h2o4gpu-buildversion${extratag}-build

echo "Docker devel - Copying files"
$DOCKER_CLI exec ${CONTAINER_NAME} bash -c 'mkdir -p repo ; cp -a /dot/. ./repo'

echo "shallow clone, and make buildinstall with ${H2O4GPU_BUILD} and ${H2O4GPU_SUFFIX}"
$DOCKER_CLI exec ${CONTAINER_NAME} bash -c "cd repo ; make ${makeopts} buildinstall H2O4GPU_BUILD=${H2O4GPU_BUILD} H2O4GPU_SUFFIX=${H2O4GPU_SUFFIX}"

echo "Docker devel - Clean local wheels and Copying wheel from docker"
rm -rf src/interface_py/dist/
$DOCKER_CLI cp -a ${CONTAINER_NAME}:/root/repo/src/interface_py/dist src/interface_py/

echo "Docker devel - Copying VERSION.txt"
mkdir -p build ; $DOCKER_CLI cp ${CONTAINER_NAME}:/root/repo/build/VERSION.txt build/

echo "Docker devel - Stopping docker"
$DOCKER_CLI stop ${CONTAINER_NAME}

echo "Docker devel - END"
