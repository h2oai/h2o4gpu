#!/bin/bash
set -e

H2O4GPU_BUILD="${H2O4GPU_BUILD:-0}"
H2O4GPU_SUFFIX="${H2O4GPU_SUFFIX:-''}"
CONTAINER_NAME="${CONTAINER_NAME:-$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)}"
makeopts="${makeopts:-}"

DOCKER_CLI='nvidia-docker'

CONDA_PKG_NAME="h2o4gpu${extratag}"

#--build-arg http_proxy=http://172.16.2.142:3128/
echo "Docker devel - BEGIN"
$DOCKER_CLI build -t opsh2oai/h2o4gpu-buildversion${extratag}-build -f Dockerfile-build --rm=false --build-arg docker_name=${dockerimage} .

$DOCKER_CLI run --init --rm --name ${CONTAINER_NAME} -d -t -u root -v `pwd`:/dot  --entrypoint=bash opsh2oai/h2o4gpu-buildversion${extratag}-build

echo "Docker devel - Copying files"
$DOCKER_CLI exec ${CONTAINER_NAME} bash -c 'mkdir -p repo ; cp -a /dot/. ./repo'

# workaround to not compile nccl every time 
echo "Docker devel - Copying nccl build artifacts"
$DOCKER_CLI exec ${CONTAINER_NAME} bash -c 'if [  $(git -C /root/nccl rev-parse HEAD) !=  $(git -C /root/repo rev-parse  @:nccl) ]; then echo "NCCL version mismatch in nccl submodule and docker file" && exit 1;  fi;'   
$DOCKER_CLI exec ${CONTAINER_NAME} bash -c 'cp -r /root/nccl/build /root/repo/nccl'

echo "shallow clone, and make buildinstall with ${H2O4GPU_BUILD} and ${H2O4GPU_SUFFIX}"
$DOCKER_CLI exec ${CONTAINER_NAME} bash -c "cd repo ; make ${makeopts} buildinstall H2O4GPU_BUILD=${H2O4GPU_BUILD} H2O4GPU_SUFFIX=${H2O4GPU_SUFFIX}"

echo "Docker devel - Clean local wheels and Copying wheel from docker"
rm -rf src/interface_py/dist/
$DOCKER_CLI cp -a ${CONTAINER_NAME}:/root/repo/src/interface_py/dist src/interface_py/

echo "Docker devel - Copying VERSION.txt"
mkdir -p build ; $DOCKER_CLI cp ${CONTAINER_NAME}:/root/repo/build/VERSION.txt build/

if [ `arch` != "ppc64le" ]; then
    echo "Docker devel - Creating conda package"
    $DOCKER_CLI exec ${CONTAINER_NAME} bash -c "mkdir -p repo/condapkgs"
    $DOCKER_CLI exec ${CONTAINER_NAME} bash -c "cd repo/src/interface_py; cat requirements_*.txt | grep -v '#' | sort | uniq > requirements_conda.txt"
    $DOCKER_CLI exec ${CONTAINER_NAME} bash -c "pushd repo/conda-recipe; sed -i 's/condapkgname/${CONDA_PKG_NAME}/g' meta.yaml; popd"
    $DOCKER_CLI exec ${CONTAINER_NAME} bash -c "pushd repo/conda-recipe; conda build --output-folder ../condapkgs  -c h2oai -c conda-forge .; popd"

    echo "Docker devel - Copying conda package"
    rm -rf condapkgs
    $DOCKER_CLI cp -a ${CONTAINER_NAME}:/root/repo/condapkgs .
fi

echo "Docker devel - Stopping docker"
$DOCKER_CLI stop ${CONTAINER_NAME}

echo "Docker devel - END"
