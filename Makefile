include Makefile_header.mk
location = $(CURDIR)/$(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))
WHERE := $(location)
$(info ** -> $(WHERE))
$(info ** ------------------------------------------------------------------ **)
NVCC := $(shell command -v nvcc 2> /dev/null)

SHELL := /bin/bash # force avoidance of dash as shell
#
# Build specific config
#
CONFIG=make/config.mk
include $(CONFIG)

VERSION=make/version.mk
include $(VERSION)

MAJOR_MINOR=$(shell echo $(BASE_VERSION) | sed 's/.*\(^[0-9][0-9]*\.[0-9][0-9]*\).*/\1/g' )

# System specific stuff
include src/config2.mk

ifdef NVCC
ifeq ($(shell test $(CUDA_MAJOR) -ge 9; echo $$?),0)
    $(warning Compiling with Cuda9 or higher)
    XGB_CUDA ?= -DGPU_COMPUTE_VER="35;52;60;61;70"
else
    $(warning Compiling with Cuda8 or lower)
    # >=52 required for kmeans for larger data of size rows/32>2^16
    XGB_CUDA ?= -DGPU_COMPUTE_VER="35;52;60;61"
endif
endif

# Location of local directory with dependencies
DEPS_DIR = deps

# Detect OS
OS := $(shell uname)
## Python has crazy ideas about os names
ifeq ($(OS), Darwin)
		PY_OS ?= "macosx"
else
		PY_OS ?= $(OS)
endif

# see if have ccache for faster compile times if no changes to file
theccache=$(shell echo `which ccache`)
ifeq ($(theccache),)
		theccacheclean=
else
		theccacheclean=$(theccache) -C
endif

RANDOM := $(shell bash -c 'echo $$RANDOM')
LOGEXT=$(RANDOM)$(shell date +'_%Y.%m.%d-%H:%M:%S')
NUMPROCS := $(shell cat /proc/cpuinfo|grep processor|wc -l)

#
# Docker image tagging
#
DOCKER_VERSION_TAG ?= "latest"

#
# Setup S3 access credentials
#
S3_CMD_LINE := aws s3

#
# BUILD_INFO setup
#
H2O4GPU_COMMIT ?= $(shell git rev-parse HEAD)
H2O4GPU_BUILD_DATE := $(shell date)
H2O4GPU_BUILD ?= "LOCAL BUILD @ $(shell git rev-parse --short HEAD) build at $(H2O4GPU_BUILD_DATE)"
H2O4GPU_SUFFIX ?= "+local_$(shell git describe --always --dirty)"


help:
	$(call inform, " -------- Build and Install ---------")
	$(call inform, "make clean           Clean all build files.")
	$(call inform, "make                 fullinstall")
	$(call inform, "make fullinstall     Clean everything then compile and install everything (for cuda9 with nccl in xgboost).")
	$(call inform, "make cpu-fullinstall Clean everything then compile and isntall everything only with CPU")
	$(call inform, "make build           Just Build the whole project.")
	$(call inform, " -------- Test ---------")
	$(call inform, "make test            Run tests.")
	$(call inform, "make testbig         Run tests for big data.")
	$(call inform, "make testperf        Run performance and accuracy tests.")
	$(call inform, "make testbigperf     Run performance and accuracy tests for big data.")
	$(call inform, " -------- Docker ---------")
	$(call inform, "make docker-build    Build inside docker and save wheel to src/interface_py/dist?/ (for cuda9 with nccl in xgboost).")
	$(call inform, "make docker-runtime  Build runtime docker and save to local path (for cuda9 with nccl in xgboost).")
	$(call inform, "make get_docker      Download runtime docker (e.g. instead of building it)")
	$(call inform, "make load_docker     Load runtime docker image")
	$(call inform, "make run_in_docker   Run jupyter notebook demo using runtime docker image already present")
	$(call inform, "make docker-runtests Run tests in docker")
	$(call inform, " -------- Pycharm Help ---------")
	$(call inform, "Example Pycharm environment flags: PYTHONPATH=/home/jon/h2o4gpu/src/interface_py:/home/jon/h2o4gpu;PYTHONUNBUFFERED=1;LD_LIBRARY_PATH=/opt/clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04//lib/:/home/jon/lib:/opt/rstudio-1.0.136/bin/:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64::/home/jon/lib/:$LD_LIBRARY_PATH;LLVM4=/opt/clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04/")
	$(call inform, "Example Pycharm working directory: /home/jon/h2o4gpu/")

sync_small_data:
	@echo "---- Synchronizing test data ----"
	mkdir -p $(DATA_DIR)
	$(S3_CMD_LINE) sync --no-sign-request "$(SMALLDATA_BUCKET)" "$(DATA_DIR)"

sync_other_data:
	@echo "---- Synchronizing data dir in test/ ----"
	mkdir -p $(DATA_DIR) && $(S3_CMD_LINE) sync "$(DATA_BUCKET)" "$(DATA_DIR)"

sync_open_data:
	@echo "---- Synchronizing sklearn and other open data in home directory ----"
	mkdir -p $(OPEN_DATA_DIR)
	$(S3_CMD_LINE) sync --no-sign-request "$(OPEN_DATA_BUCKET)" "$(OPEN_DATA_DIR)"

default: fullinstall

#########################################

update_submodule:
	echo ADD UPDATE SUBMODULE HERE

cpp:
	(mkdir -p build; cd build; cmake ../; make -j)

py: apply-sklearn_simple build/VERSION.txt
	$(MAKE) -j all -C src/interface_py

pylint:
	$(MAKE) pylint -C src/interface_py

fullpy: apply-sklearn_simple pylint

pyinstall:
	$(MAKE) -j install -C src/interface_py


##############################################

alldeps-nccl-cuda8: deps_fetch alldeps_install-nccl-cuda8
alldeps-nonccl-cuda8: deps_fetch alldeps_install-nonccl-cuda8
alldeps-nccl-cuda9: deps_fetch alldeps_install-nccl-cuda9
alldeps-nonccl-cuda9: deps_fetch alldeps_install-nonccl-cuda9
alldeps-cpuonly: deps_fetch alldeps_install-cpuonly

clean: cleanbuild deps_clean xgboost_clean py3nvml_clean
	-rm -rf ./build
	-rm -rf ./results/ ./tmp/

cleanbuild: cleanpy

cleanpy:
	$(MAKE) -j clean -C src/interface_py

xgboost_clean:
	-pip uninstall -y xgboost
	rm -rf xgboost/build/

buildquick: cpp py
build: update_submodule buildquick
buildnocpp: update_submodule cleanpy py # avoid cpp

install: pyinstall

### for direct building of xgboost
# https://xgboost.readthedocs.io/en/latest/build.html
libxgboost-nccl-local:
	cd xgboost ; make -f Makefile2 libxgboost
libxgboost-nonccl-local:
	cd xgboost ; make -f Makefile2 libxgboost2
libxgboost-cpu-local:
	cd xgboost ; make -f Makefile2 libxgboost-cpu

apply-xgboost-nccl-local: libxgboost-nccl-local pipxgboost
apply-xgboost-nonccl-local: libxgboost-nonccl-local pipxgboost
apply-xgboost-cpu-local: libxgboost-cpu-local pipxgboost

pipxgboost:
	@echo "----- pip install xgboost built locally -----"
	cd xgboost/python-package/dist && pip install xgboost-0.7-py3-none-any.whl --upgrade --target ../

alldeps-nccl-local: deps_fetch alldeps-install-nccl-local
alldeps-nonccl-local: deps_fetch alldeps-install-nonccl-local
alldeps-cpu-local: deps_fetch alldeps-install-cpu-local

# lib for sklearn because don't want to fully apply yet
alldeps-install-nccl-local: deps_install apply-xgboost-nccl-local apply_py3nvml libsklearn
alldeps-install-nonccl-local: deps_install apply-xgboost-nonccl-local apply_py3nvml libsklearn
alldeps-install-cpu-local: deps_install apply-xgboost-cpu-local apply_py3nvml libsklearn
alldeps_install-cpuonly: deps_install apply-xgboost-cpu-local apply_py3nvml libsklearn install_daal_x86_64

##### dependencies
deps_clean:
	@echo "----- Cleaning deps -----"
	rm -rf "$(DEPS_DIR)"
	# sometimes --upgrade leaves extra packages around
	cat requirements_buildonly.txt requirements_runtime.txt requirements_runtime_demos.txt > requirements.txt
	sed 's/==.*//g' requirements.txt|grep -v "#" > requirements_plain.txt
	-xargs -a requirements_plain.txt -n 1 -P $(NUMPROCS) pip uninstall -y
	rm -rf requirements_plain.txt requirements.txt

deps_fetch:
	@echo "---- Fetch dependencies ---- "
	bash scripts/gitshallow_submodules.sh
	git submodule update

deps_install:
	@echo "---- Install dependencies ----"
	#-xargs -a requirements.txt -n 1 -P 1 pip install --upgrade
	easy_install pip
	easy_install setuptools
	cat requirements_buildonly.txt requirements_runtime.txt > requirements.txt
	pip install -r requirements.txt --upgrade
	rm -rf requirements.txt
	bash scripts/install_r_deps.sh
	# issue with their package, have to do this here (still fails sometimes, so remove)
#	pip install sphinxcontrib-osexample

# lib for sklearn because don't want to fully apply yet
alldeps_install-nccl-cuda8: deps_install apply-xgboost-nccl-cuda8 apply_py3nvml libsklearn
alldeps_install-nonccl-cuda8: deps_install apply-xgboost-nonccl-cuda8 apply_py3nvml libsklearn
alldeps_install-nccl-cuda9: deps_install apply-xgboost-nccl-cuda9 apply_py3nvml libsklearn
alldeps_install-nonccl-cuda9: deps_install apply-xgboost-nonccl-cuda9 apply_py3nvml libsklearn

fullinstall: fullinstall-nccl-cuda9
fullinstalllocal: fullinstall-nccl-local
cpu-fullinstall: fullinstall-cpuonly

fullinstall-nccl-local: clean alldeps-nccl-local build install
	mkdir -p src/interface_py/dist-nccl-local/ && mv src/interface_py/dist/*.whl src/interface_py/dist-nccl-local/
fullinstall-nonccl-local: clean alldeps-nonccl-local build install
	mkdir -p src/interface_py/dist-nonccl-local/ && mv src/interface_py/dist/*.whl src/interface_py/dist-nonccl-local/

fullinstall-nccl-cuda8: clean alldeps-nccl-cuda8 build install
	mkdir -p src/interface_py/dist1/ && mv src/interface_py/dist/*.whl src/interface_py/dist1/

fullinstall-nonccl-cuda8: clean alldeps-nonccl-cuda8 build install
	mkdir -p src/interface_py/dist2/ && mv src/interface_py/dist/*.whl src/interface_py/dist2/

fullinstall-nccl-cuda9: clean alldeps-nccl-cuda9 build install
	mkdir -p src/interface_py/dist4/ && mv src/interface_py/dist/*.whl src/interface_py/dist4/

fullinstall-nonccl-cuda9: clean alldeps-nonccl-cuda9 build install
	mkdir -p src/interface_py/dist3/ && mv src/interface_py/dist/*.whl src/interface_py/dist3/

fullinstall-cpuonly: clean alldeps-cpuonly build install
	mkdir -p src/interface_py/dist-cpuonly-local/ && mv src/interface_py/dist/*.whl src/interface_py/dist-cpuonly-local/

####################################################
# Docker stuff

# default for docker is nccl-cuda9
docker-build: docker-build-nccl-cuda9
docker-runtime: docker-runtime-nccl-cuda9
docker-runtests: docker-runtests-nccl-cuda9
get_docker: get_docker-nccl-cuda9
load_docker: docker-runtime-nccl-cuda9-load
run_in_docker: run_in_docker-nccl-cuda9


############### CUDA9

docker-build-nccl-cuda9:
	@echo "+-- Building Wheel in Docker (-nccl-cuda9) --+"
	rm -rf src/interface_py/dist/*.whl ; rm -rf src/interface_py/dist4/*.whl
	export CONTAINER_NAME="localmake-build" ;\
	export versionTag=$(BASE_VERSION) ;\
	export extratag="-nccl-cuda9" ;\
	export dockerimage="nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04" ;\
	export H2O4GPU_BUILD="" ;\
	export H2O4GPU_SUFFIX="" ;\
	export makeopts="" ;\
	export dist="dist4" ;\
	bash scripts/make-docker-devel.sh

docker-runtime-nccl-cuda9:
	@echo "+--Building Runtime Docker Image Part 2 (-nccl-cuda9) --+"
	export CONTAINER_NAME="localmake-runtime" ;\
	export versionTag=$(BASE_VERSION) ;\
	export extratag="-nccl-cuda9" ;\
	export encodedFullVersionTag=$(BASE_VERSION) ;\
	export fullVersionTag=$(BASE_VERSION) ;\
	export buckettype="releases/bleeding-edge" ;\
	export dockerimage="nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04" ;\
	bash scripts/make-docker-runtime.sh

.PHONY: docker-runtime-nccl-cuda9-run

docker-runtime-nccl-cuda9-run:
	@echo "+-Running Docker Runtime Image (-nccl-cuda9) --+"
	export CONTAINER_NAME="localmake-runtime-run" ;\
	export versionTag=$(BASE_VERSION) ;\
	export extratag="-nccl-cuda9" ;\
	export encodedFullVersionTag=$(BASE_VERSION) ;\
	export fullVersionTag=$(BASE_VERSION) ;\
	export buckettype="releases/bleeding-edge" ;\
	export dockerimage="nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04" ;\
	nvidia-docker run --init --rm --name $${CONTAINER_NAME} -d -t -u `id -u`:`id -g` --entrypoint=bash opsh2oai/h2o4gpu-$${versionTag}$${extratag}-runtime:latest

docker-runtests-nccl-cuda9:
	@echo "+-- Run tests in docker (-nccl-cuda9) --+"
	export CONTAINER_NAME="localmake-runtests" ;\
	export extratag="-nccl-cuda9" ;\
	export dockerimage="nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04" ;\
	export dist="dist4" ;\
	export target="dotest" ;\
	bash scripts/make-docker-runtests.sh

get_docker-nccl-cuda9:
	wget https://s3.amazonaws.com/h2o-release/h2o4gpu/releases/bleeding-edge/ai/h2o/h2o4gpu/$(MAJOR_MINOR)-nccl-cuda9/h2o4gpu-$(BASE_VERSION)-nccl-cuda9-runtime.tar.bz2

docker-runtime-nccl-cuda9-load:
	pbzip2 -dc h2o4gpu-$(BASE_VERSION)-nccl-cuda9-runtime.tar.bz2 | nvidia-docker load

run_in_docker-nccl-cuda9:
	-mkdir -p log ; nvidia-docker run --name localhost --rm -p 8888:8888 -u `id -u`:`id -g` -v `pwd`/log:/log --entrypoint=./run.sh opsh2oai/h2o4gpu-$(BASE_VERSION)-nccl-cuda9-runtime &
	-find log -name jupyter* -type f -printf '%T@ %p\n' | sort -k1 -n | awk '{print $2}' | tail -1 | xargs cat | grep token | grep http | grep -v NotebookApp


############### CPU
docker-build-cpu:
	@echo "+-- Building Wheel in Docker (-cpu) --+"
	rm -rf src/interface_py/dist/*.whl ; rm -rf src/interface_py/dist4/*.whl
	export CONTAINER_NAME="localmake-build" ;\
	export versionTag=$(BASE_VERSION) ;\
	export extratag="-cpu" ;\
	export dockerimage="ubuntu:16.04" ;\
	export H2O4GPU_BUILD="" ;\
	export H2O4GPU_SUFFIX="" ;\
	export makeopts="" ;\
	export dist="dist8" ;\
	bash scripts/make-docker-devel.sh

docker-runtime-cpu:
	@echo "+--Building Runtime Docker Image Part 2 (-cpu) --+"
	export CONTAINER_NAME="localmake-runtime" ;\
	export versionTag=$(BASE_VERSION) ;\
	export extratag="-cpu" ;\
	export encodedFullVersionTag=$(BASE_VERSION) ;\
	export fullVersionTag=$(BASE_VERSION) ;\
	export buckettype="releases/bleeding-edge" ;\
	export dockerimage="ubuntu:16.04" ;\
	bash scripts/make-docker-runtime.sh

docker-runtime-cpu-run:
	@echo "+-Running Docker Runtime Image (-nccl-cuda9) --+"
	export CONTAINER_NAME="localmake-runtime-run" ;\
	export versionTag=$(BASE_VERSION) ;\
	export extratag="-cpu" ;\
	export encodedFullVersionTag=$(BASE_VERSION) ;\
	export fullVersionTag=$(BASE_VERSION) ;\
	export buckettype="releases/bleeding-edge" ;\
	export dockerimage="ubuntu:16.04" ;\
	docker run --init --rm --name $${CONTAINER_NAME} -d -t -u `id -u`:`id -g` --entrypoint=bash opsh2oai/h2o4gpu-$${versionTag}$${extratag}-runtime:latest

docker-runtests-cpu:
	@echo "+-- Run tests in docker (-nccl-cuda9) --+"
	export CONTAINER_NAME="localmake-runtests" ;\
	export extratag="-cpu" ;\
	export dockerimage="ubuntu:16.04" ;\
	export dist="dist4" ;\
	export target="dotest" ;\
	bash scripts/make-docker-runtests.sh

get_docker-cpu:
	wget https://s3.amazonaws.com/h2o-release/h2o4gpu/releases/bleeding-edge/ai/h2o/h2o4gpu/$(MAJOR_MINOR)-cpu/h2o4gpu-$(BASE_VERSION)-cpu-runtime.tar.bz2

docker-runtime-cpu-load:
	pbzip2 -dc h2o4gpu-$(BASE_VERSION)-cpu-runtime.tar.bz2 | docker load

run_in_docker-cpu:
	-mkdir -p log ; docker run --name localhost --rm -p 8888:8888 -u `id -u`:`id -g` -v `pwd`/log:/log --entrypoint=./run.sh opsh2oai/h2o4gpu-$(BASE_VERSION)-cpu-runtime &
	-find log -name jupyter* -type f -printf '%T@ %p\n' | sort -k1 -n | awk '{print $2}' | tail -1 | xargs cat | grep token | grep http | grep -v NotebookApp


######### CUDA8 (copy/paste above, and then replace cuda9 -> cuda8 and cuda:9.0-cudnn7 -> cuda:8.0-cudnn5 and dist4->dist1)

docker-build-nccl-cuda8:
	@echo "+-- Building Wheel in Docker (-nccl-cuda8) --+"
	rm -rf src/interface_py/dist/*.whl
	export CONTAINER_NAME="localmake-build" ;\
	export versionTag=$(BASE_VERSION) ;\
	export extratag="-nccl-cuda8" ;\
	export dockerimage="nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04" ;\
	export H2O4GPU_BUILD="" ;\
	export H2O4GPU_SUFFIX="" ;\
	export makeopts="" ;\
	export dist="dist1" ;\
	bash scripts/make-docker-devel.sh

docker-runtime-nccl-cuda8:
	@echo "+--Building Runtime Docker Image Part 2 (-nccl-cuda8) --+"
	export CONTAINER_NAME="localmake-runtime" ;\
	export versionTag=$(BASE_VERSION) ;\
	export extratag="-nccl-cuda8" ;\
	export encodedFullVersionTag=$(BASE_VERSION) ;\
	export fullVersionTag=$(BASE_VERSION) ;\
	export buckettype="releases/bleeding-edge" ;\
	export dockerimage="nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04" ;\
	bash scripts/make-docker-runtime.sh

docker-runtime-nccl-cuda8-load:
	pbzip2 -dc h2o4gpu-$(BASE_VERSION)-nccl-cuda8-runtime.tar.bz2 | nvidia-docker load

.PHONY: docker-runtime-nccl-cuda8-run

docker-runtime-nccl-cuda8-run:
	@echo "+-Running Docker Runtime Image (-nccl-cuda8) --+"
	export CONTAINER_NAME="localmake-runtime-run" ;\
	export versionTag=$(BASE_VERSION) ;\
	export extratag="-nccl-cuda8" ;\
	export encodedFullVersionTag=$(BASE_VERSION) ;\
	export fullVersionTag=$(BASE_VERSION) ;\
	export buckettype="releases/bleeding-edge" ;\
	export dockerimage="nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04" ;\
	nvidia-docker run --init --rm --name $${CONTAINER_NAME} -d -t -u `id -u`:`id -g` --entrypoint=bash opsh2oai/h2o4gpu-$${versionTag}$${extratag}-runtime:latest

docker-runtests-nccl-cuda8:
	@echo "+-- Run tests in docker (-nccl-cuda8) --+"
	export CONTAINER_NAME="localmake-runtests" ;\
	export extratag="-nccl-cuda8" ;\
	export dockerimage="nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04" ;\
	export dist="dist1" ;\
	export target="dotest" ;\
	bash scripts/make-docker-runtests.sh

get_docker-nccl-cuda8:
	wget https://s3.amazonaws.com/h2o-release/h2o4gpu/releases/bleeding-edge/ai/h2o/h2o4gpu/$(MAJOR_MINOR)-nccl-cuda8/h2o4gpu-$(BASE_VERSION)-nccl-cuda8-runtime.tar.bz2

run_in_docker-nccl-cuda8:
	mkdir -p log ; nvidia-docker run --name localhost --rm -p 8888:8888 -u `id -u`:`id -g` -v `pwd`/log:/log --entrypoint=./run.sh opsh2oai/h2o4gpu-$(BASE_VERSION)-nccl-cuda8-runtime &
	find log -name jupyter* | xargs cat | grep token | grep http | grep -v NotebookApp

# uses https://github.com/Azure/fast_retraining
testxgboost: # liblightgbm (assumes one installs lightgdm yourself or run make liblightgbm)
	bash testsxgboost/runtestxgboost.sh
	bash testsxgboost/extracttestxgboost.sh
	bash tests_open/showresults.sh # same for all tests

# install daal
install_daal_x86_64:
	@echo "----- Install Daal Python library -----"
	bash scripts/daal/install_daal_locally.sh

################

# http://developer2.download.nvidia.com/compute/cuda/9.0/secure/rc/docs/sidebar/CUDA_Quick_Start_Guide.pdf?_ZyOB0PlGZzBUluXp3FtoWC-LMsTsc5H6SxIaU0i9pGNyWzZCgE-mhnAg2m66Nc3WMDvxWvvQWsXGMqr1hUliGOZvoothMTVnDe12dQQgxwS4Asjoz8XiOvPYOjV6yVQtkFhvDztUlJbNSD4srPWUU2-XegCRFII8_FIpxXERaWV
libcuda9:
	# wget https://developer.nvidia.com/compute/cuda/9.0/rc/local_installers/cuda-repo-ubuntu1604-9-0-local-rc_9.0.103-1_amd64-deb
	sudo dpkg --install cuda-repo-ubuntu1604-9-0-local-rc_9.0.103-1_amd64.deb
	# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
	sudo apt-key add 7fa2af80.pub
	sudo apt-get update
	sudo apt-get install cuda

# http://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html
libnccl2:
	# cuda8 nccl2
	#wget https://developer.nvidia.com/compute/machine-learning/nccl/secure/v2.0/prod/nccl-repo-ubuntu1604-2.0.5-ga-cuda8.0_2-1_amd64-deb
	# cuda9 nccl2
	# wget https://developer.nvidia.com/compute/machine-learning/nccl/secure/v2.0/prod/nccl-repo-ubuntu1604-2.0.5-ga-cuda9.0_2-1_amd64-deb
	sudo dpkg -i nccl-repo-ubuntu1604-2.0.5-ga-cuda9.0_2-1_amd64.deb
	sudo apt update
	sudo apt-key add /var/nccl-repo-2.0.5-ga-cuda9.0/7fa2af80.pub
	sudo apt install libnccl2 libnccl-dev

apply-xgboost-nccl-cuda8: apply-xgboost-nccl-local    #pipxgboost-nccl-cuda8
apply-xgboost-nonccl-cuda8: apply-xgboost-nonccl-local #pipxgboost-nonccl-cuda8
apply-xgboost-nccl-cuda9: apply-xgboost-nccl-local #pipxgboost-nccl-cuda9
apply-xgboost-nonccl-cuda9: apply-xgboost-nonccl-local #pipxgboost-nonccl-cuda9

pipxgboost-nccl-cuda8: pipxgboost
	@echo "----- pip install xgboost-nccl-cuda8 from S3 -----"
	mkdir -p xgboost/python-package/dist ; cd xgboost/python-package/dist && pip install https://s3.amazonaws.com/artifacts.h2o.ai/releases/bleeding-edge/ai/h2o/xgboost/0.7-nccl-cuda8/xgboost-0.7-py3-none-any.whl --upgrade --target ../
pipxgboost-nonccl-cuda8: pipxgboost
	@echo "----- pip install xgboost-nonccl-cuda8 from S3 -----"
	mkdir -p xgboost/python-package/dist ; cd xgboost/python-package/dist && pip install https://s3.amazonaws.com/artifacts.h2o.ai/releases/bleeding-edge/ai/h2o/xgboost/0.7-nonccl-cuda8/xgboost-0.7-py3-none-any.whl --upgrade --target ../
pipxgboost-nccl-cuda9: pipxgboost
	@echo "----- pip install xgboost-nccl-cuda9 from S3 -----"
	mkdir -p xgboost/python-package/dist ; cd xgboost/python-package/dist && pip install https://s3.amazonaws.com/artifacts.h2o.ai/releases/bleeding-edge/ai/h2o/xgboost/0.7-nccl-cuda9/xgboost-0.7-py3-none-any.whl --upgrade --target ../
pipxgboost-nonccl-cuda9: pipxgboost
	@echo "----- pip install xgboost-nonccl-cuda9 from S3 -----"
	mkdir -p xgboost/python-package/dist ; cd xgboost/python-package/dist && pip install https://s3.amazonaws.com/artifacts.h2o.ai/releases/bleeding-edge/ai/h2o/xgboost/0.7-nonccl-cuda9/xgboost-0.7-py3-none-any.whl --upgrade --target ../


py3nvml_clean:
	-pip uninstall -y py3nvml

apply_py3nvml:
	mkdir -p py3nvml ; cd py3nvml # ; pip install -e git+https://github.com/fbcotter/py3nvml#egg=py3nvml --upgrade --root=.


liblightgbm: # only done if user directly requests, never an explicit dependency
	echo "See https://github.com/Microsoft/LightGBM/wiki/Installation-Guide#with-gpu-support for details"
	echo "sudo apt-get install libboost-dev libboost-system-dev libboost-filesystem-dev cmake"
	rm -rf LightGBM ; result=`git clone --recursive https://github.com/Microsoft/LightGBM`
	cd LightGBM && mkdir build ; cd build && cmake .. -DUSE_GPU=1 -DOpenCL_LIBRARY=$(CUDA_HOME)/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=$(CUDA_HOME)/include/ && make -j && cd ../python-package ; python setup.py install --precompile --gpu && cd ../ && pip install arff tqdm keras runipy h5py --upgrade

libsklearn:	# assume already submodule gets sklearn
	@echo "----- Make sklearn wheel -----"
	bash scripts/prepare_sklearn.sh # repeated calls don't hurt
	rm -rf sklearn && mkdir -p sklearn && cd scikit-learn && python setup.py sdist bdist_wheel

apply-sklearn: libsklearn apply-sklearn_simple

apply-sklearn_simple:
    #	bash ./scripts/apply_sklearn.sh
    ## apply sklearn
	bash ./scripts/apply_sklearn_pipinstall.sh
    ## link-up recursively
	bash ./scripts/apply_sklearn_link.sh
    # handle base __init__.py file appending
	bash ./scripts/apply_sklearn_initmerge.sh

apply-sklearn_pipinstall:
	bash ./scripts/apply_sklearn_pipinstall.sh

apply-sklearn_link:
	bash ./scripts/apply_sklearn_link.sh

apply-sklearn_initmerge:
	bash ./scripts/apply_sklearn_initmerge.sh

#################### Jenkins specific

######### h2o.ai systems
# for nccl cuda8 build
fullinstalljenkins-nccl-cuda8: mrproper fullinstall-nccl-cuda8
fullinstalljenkins-nonccl-cuda8: mrproper fullinstall-nonccl-cuda8
fullinstalljenkins-nccl-cuda9: mrproper fullinstall-nccl-cuda9
fullinstalljenkins-nonccl-cuda9: mrproper fullinstall-nonccl-cuda9
fullinstalljenkins-cpu: mrproper fullinstall-cpu-local

# for nccl cuda9 build benchmark
fullinstalljenkins-nccl-cuda9-benchmark: mrproper clean alldeps-nccl-cuda9 build install
	mkdir -p src/interface_py/dist6/ && mv src/interface_py/dist/*.whl src/interface_py/dist6/

########## AWS
# for nccl cuda9 build aws build/test
fullinstalljenkins-nccl-cuda9-aws1: mrproper clean alldeps-nccl-cuda9 build install
	mkdir -p src/interface_py/dist5/ && mv src/interface_py/dist/*.whl src/interface_py/dist5/

# for nccl cuda9 build benchmark on aws1
fullinstalljenkins-nccl-cuda9-aws1-benchmark: mrproper clean alldeps-nccl-cuda9 build install
	mkdir -p src/interface_py/dist7/ && mv src/interface_py/dist/*.whl src/interface_py/dist7/

.PHONY: mrproper
mrproper: clean
	@echo "----- Cleaning properly -----"
	git clean -f -d -x


##################

#WIP
dotestdemos:
	rm -rf ./tmp/
	mkdir -p ./tmp/
	bash scripts/convert_ipynb2py.sh
    # can't do -n auto due to limits on GPU memory
	#pytest -s --verbose --durations=10 -n 1 --fulltrace --full-trace --junit-xml=build/test-reports/h2o4gpu-test.xml examples/py 2> ./tmp/h2o4gpu-examplespy.$(LOGEXT).log
	-pip install pytest-ipynb # can't put in requirements since problem with jenkins and runipy
	py.test -v -s examples/py 2> ./tmp/h2o4gpu-examplespy.$(LOGEXT).log


dotest:
	rm -rf ./tmp/
	mkdir -p ./tmp/
  # can't do -n auto due to limits on GPU memory
	pytest -s --verbose --durations=10 -n 3 --fulltrace --full-trace --junit-xml=build/test-reports/h2o4gpu-test.xml tests_open 2> ./tmp/h2o4gpu-test.$(LOGEXT).log
	# Test R package when appropriate
	bash scripts/test_r_pkg.sh

dotestfast:
	rm -rf ./tmp/
	mkdir -p ./tmp/
    # can't do -n auto due to limits on GPU memory
	pytest -s --verbose --durations=10 -n 3 --fulltrace --full-trace --junit-xml=build/test-reports/h2o4gpu-testfast1.xml tests_open/glm/test_glm_simple.py 2> ./tmp/h2o4gpu-testfast1.$(LOGEXT).log
	pytest -s --verbose --durations=10 -n 3 --fulltrace --full-trace --junit-xml=build/test-reports/h2o4gpu-testfast2.xml tests_open/gbm/test_xgb_sklearn_wrapper.py 2> ./tmp/h2o4gpu-testfast2.$(LOGEXT).log
	pytest -s --verbose --durations=10 -n 3 --fulltrace --full-trace --junit-xml=build/test-reports/h2o4gpu-testfast3.xml tests_open/svd/test_tsvd.py 2> ./tmp/h2o4gpu-testfast3.$(LOGEXT).log
	pytest -s --verbose --durations=10 -n 3 --fulltrace --full-trace --junit-xml=build/test-reports/h2o4gpu-testfast4.xml tests_open/kmeans/test_kmeans.py 2> ./tmp/h2o4gpu-testfast4.$(LOGEXT).log

dotestfast_nonccl:
	rm -rf ./tmp/
	mkdir -p ./tmp/
	# can't do -n auto due to limits on GPU memory
	pytest -s --verbose --durations=10 -n 3 --fulltrace --full-trace --junit-xml=build/test-reports/h2o4gpu-testfast1.xml tests_open/glm/test_glm_simple.py 2> ./tmp/h2o4gpu-testfast1.$(LOGEXT).log
	pytest -s --verbose --durations=10 -n 3 --fulltrace --full-trace --junit-xml=build/test-reports/h2o4gpu-testfast3.xml tests_open/svd/test_tsvd.py 2> ./tmp/h2o4gpu-testfast3.$(LOGEXT).log
	pytest -s --verbose --durations=10 -n 3 --fulltrace --full-trace --junit-xml=build/test-reports/h2o4gpu-testfast4.xml tests_open/kmeans/test_kmeans.py 2> ./tmp/h2o4gpu-testfast4.$(LOGEXT).log

dotestsmall:
	rm -rf ./tmp/
	rm -rf build/test-reports 2>/dev/null
	mkdir -p ./tmp/
    # can't do -n auto due to limits on GPU memory
	pytest -s --verbose --durations=10 -n 3 --fulltrace --full-trace --junit-xml=build/test-reports/h2o4gpu-testsmall.xml tests_small 2> ./tmp/h2o4gpu-testsmall.$(LOGEXT).log

dotestbig:
	mkdir -p ./tmp/
	pytest -s --verbose --durations=10 -n 1 --fulltrace --full-trace --junit-xml=build/test-reports/h2o4gpu-testbig.xml tests_big 2> ./tmp/h2o4gpu-testbig.$(LOGEXT).log

#####################

dotestperf:
	mkdir -p ./tmp/
	-CHECKPERFORMANCE=1 DISABLEPYTEST=1 pytest -s --verbose --durations=10 -n 1 --fulltrace --full-trace --junit-xml=build/test-reports/h2o4gpu-test.xml tests_open 2> ./tmp/h2o4gpu-testperf.$(LOGEXT).log
	bash tests_open/showresults.sh &> ./tmp/h2o4gpu-testperf-results.$(LOGEXT).log

dotestsmallperf:
	mkdir -p ./tmp/
	-CHECKPERFORMANCE=1 DISABLEPYTEST=1 pytest -s --verbose --durations=10 -n 1 --fulltrace --full-trace --junit-xml=build/test-reports/h2o4gpu-testsmallperf.xml tests_small 2> ./tmp/h2o4gpu-testsmallperf.$(LOGEXT).log
	bash tests_open/showresults.sh &> ./tmp/h2o4gpu-testsmallperf-results.$(LOGEXT).log

dotestbigperf:
	mkdir -p ./tmp/
	-CHECKPERFORMANCE=1 DISABLEPYTEST=1 pytest -s --verbose --durations=10 -n 1 --fulltrace --full-trace --junit-xml=build/test-reports/h2o4gpu-testbigperf.xml tests_big 2> ./tmp/h2o4gpu-testbigperf.$(LOGEXT).log
	bash tests_open/showresults.sh  &> ./tmp/h2o4gpu-testbigperf-results.$(LOGEXT).log # still just references results directory in base path

######################### use python instead of pytest (required in some cases if pytest leads to hang)

dotestperfpython:
	mkdir -p ./tmp/
	-bash tests_open/getresults.sh $(LOGEXT)
	bash tests_open/showresults.sh

dotestbigperfpython:
	mkdir -p ./tmp/
	-bash testsbig/getresultsbig.sh $(LOGEXT)
	bash tests_open/showresults.sh # still just references results directory in base path

################### H2O.ai public tests for pass/fail

testdemos: dotestdemos

test: buildquick dotest

testquick: dotest

################ H2O.ai public tests for performance

testperf: buildquick dotestperf # faster if also run sync_open_data before doing this test

################### H2O.ai private tests for pass/fail

testsmall: buildquick sync_open_data sync_other_data dotestsmall

testsmallquick: dotestsmall

testbig: buildquick sync_open_data sync_other_data dotestbig

testbigquick: dotestbig

################ H2O.ai private tests for performance

testsmallperf: buildquick sync_open_data sync_other_data dotestsmallperf

testbigperf: buildquick sync_open_data sync_other_data dotestbigperf

testsmallperfquick: dotestsmallperf

testbigperfquick: dotestbigperf

#################### CPP Tests

test_cpp:
	$(MAKE) -j test_cpp -C src/

clean_test_cpp:
	$(MAKE) -j clean_cpp_tests -C src/

#################### Build info

# Generate local build info
src/interface_py/h2o4gpu/BUILD_INFO.txt:
	@echo "build=\"$(H2O4GPU_BUILD)\"" > $@
	@echo "suffix=\"$(H2O4GPU_SUFFIX)\"" >> $@
	@echo "commit=\"$(H2O4GPU_COMMIT)\"" >> $@
	@echo "branch=\"`git rev-parse HEAD | git branch -a --contains | grep -v detached | sed -e 's~remotes/origin/~~g' -e 's~^ *~~' | sort | uniq | tr '*\n' ' '`\"" >> $@
	@echo "describe=\"`git describe --always --dirty`\"" >> $@
	@echo "build_os=\"`uname -a`\"" >> $@
	@echo "build_machine=\"`hostname`\"" >> $@
	@echo "build_date=\"$(H2O4GPU_BUILD_DATE)\"" >> $@
	@echo "build_user=\"`id -u -n`\"" >> $@
	@echo "base_version=\"$(BASE_VERSION)\"" >> $@
	@echo "h2o4gpu_commit=\"$(H2O4GPU_COMMIT)\"" >> $@

build/VERSION.txt: src/interface_py/h2o4gpu/BUILD_INFO.txt
	@mkdir -p build
	cd src/interface_py/; python setup.py --version > ../../build/VERSION.txt

.PHONY: base_version
base_version:
	@echo $(BASE_VERSION)

# Refresh the build info only locally, let Jenkins to generate its own
ifeq ($(CI),)
src/interface_py/h2o4gpu/BUILD_INFO.txt: .ALWAYS_REBUILD
endif

Jenkinsfiles:
	bash scripts/make_jenkinsfiles.sh

#----------------------------------------------------------------------
# CentOS 7 build API BEGIN
#
# Summary
#
#     command:  make centos7_cuda8_in_docker
#     output:   dist/x86_64-centos7-cuda8/h2o4gpu*.whl
#
#     command:  make centos7_cuda9_in_docker
#     output:   dist/x86_64-centos7-cuda9/h2o4gpu*.whl
#
# Details
#
#     This is only supported in a docker environment.
#
#     The 'centos7' make target does the actual work.
#
#     The 'centos7_cudaN_in_docker' make target sets up the docker environment
#     and then invokes the work inside that environment.
#
#     The build output is put in the 'dist' directory in h2o4gpu level.
#----------------------------------------------------------------------

DIST_DIR = dist

ARCH := $(shell arch)
PLATFORM = $(ARCH)-centos7-cuda$(MY_CUDA_VERSION)

CONTAINER_NAME_SUFFIX ?= -$(USER)
CONTAINER_NAME ?= opsh2oai/dai-h2o4gpu$(CONTAINER_NAME_SUFFIX)

PROJECT_VERSION := $(BASE_VERSION)
BRANCH_NAME ?= $(shell git rev-parse --abbrev-ref HEAD)
BRANCH_NAME_SUFFIX = +$(BRANCH_NAME)
BUILD_NUM ?= local
BUILD_NUM_SUFFIX = .$(BUILD_NUM)
VERSION = $(PROJECT_VERSION)$(BRANCH_NAME_SUFFIX)$(BUILD_NUM_SUFFIX)
CONTAINER_TAG := $(shell echo $(VERSION) | sed 's/+/-/g')

CONTAINER_NAME_TAG = $(CONTAINER_NAME):$(CONTAINER_TAG)

ARCH_SUBST = undefined
FROM_SUBST = undefined
ifeq ($(ARCH),x86_64)
    FROM_SUBST = nvidia\/cuda:$(MY_CUDA_VERSION)-cudnn$(MY_CUDNN_VERSION)-devel-centos7
    ARCH_SUBST = $(ARCH)
endif
ifeq ($(ARCH),ppc64le)
    FROM_SUBST = nvidia\/cuda-ppc64le:$(MY_CUDA_VERSION)-cudnn$(MY_CUDNN_VERSION)-devel-centos7
    ARCH_SUBST = $(ARCH)
endif

fullinstalljenkins-nonccl-cuda8-centos: mrproper centos7_in_docker

Dockerfile-build-centos7.$(PLATFORM): Dockerfile-build-centos7.in
	cat $< | sed 's/FROM_SUBST/$(FROM_SUBST)/'g | sed 's/ARCH_SUBST/$(ARCH_SUBST)/g' > $@

centos7_cuda8_in_docker: MY_CUDA_VERSION=8.0
centos7_cuda8_in_docker: MY_CUDNN_VERSION=5
centos7_cuda8_in_docker:
	$(MAKE) MY_CUDA_VERSION=$(MY_CUDA_VERSION) MY_CUDNN_VERSION=$(MY_CUDNN_VERSION) centos7_in_docker_impl

centos7_cuda9_in_docker: MY_CUDA_VERSION=9.0
centos7_cuda9_in_docker: MY_CUDNN_VERSION=7
centos7_cuda9_in_docker:
	$(MAKE) MY_CUDA_VERSION=$(MY_CUDA_VERSION) MY_CUDNN_VERSION=$(MY_CUDNN_VERSION) centos7_in_docker_impl

centos7_cuda91_in_docker: MY_CUDA_VERSION=9.1
centos7_cuda91_in_docker: MY_CUDNN_VERSION=7
centos7_cuda91_in_docker:
	$(MAKE) MY_CUDA_VERSION=$(MY_CUDA_VERSION) MY_CUDNN_VERSION=$(MY_CUDNN_VERSION) centos7_in_docker_impl

centos7_in_docker_impl: Dockerfile-build-centos7.$(PLATFORM)
	mkdir -p $(DIST_DIR)/$(PLATFORM)
	docker build \
		-t $(CONTAINER_NAME_TAG) \
		-f Dockerfile-build-centos7.$(PLATFORM) \
		.
	docker run \
		--rm \
		--init \
		-u `id -u`:`id -g` \
		-v `pwd`:/dot \
		-w /dot \
		--entrypoint /bin/bash \
		-e "MY_CUDA_VERSION=$(MY_CUDA_VERSION)" \
		-e "MY_CUDNN_VERSION=$(MY_CUDNN_VERSION)" \
		$(CONTAINER_NAME_TAG) \
		-c 'make centos7'
	echo $(VERSION) > $(DIST_DIR)/$(PLATFORM)/VERSION.txt

centos7_setup:
	rm -fr /tmp/build
	cp -a /dot/. /tmp/build
	sed -i 's/cmake/# cmake/' /tmp/build/requirements_buildonly.txt

centos7_build:
	(cd /tmp/build && \
	 IFLAGS="-I/usr/include/openblas" \
	 OPENBLAS_PREFIX="open" \
	 USEPARALLEL=0 \
	 $(MAKE) \
		deps_fetch \
		apply-xgboost-nonccl-local \
		apply_py3nvml \
		libsklearn \
		build)
	mkdir -p dist/$(PLATFORM)
	cp /tmp/build/src/interface_py/dist/h2o4gpu*.whl dist/$(PLATFORM)
	chmod -R o+rwx dist/$(PLATFORM)

centos7:
	$(MAKE) centos7_setup
	$(MAKE) centos7_build

# Note:  We don't actually need to run mrproper in docker (as root) because
#        the build step runs as the user.  But keep the API for consistency.
mrproper_in_docker:
	git clean -f -d -x

printvars: MY_CUDA_VERSION=8.0
printvars: MY_CUDNN_VERSION=5
printvars:
	@echo $(PLATFORM)
	@echo $(PROJECT_VERSION)
	@echo $(VERSION)
	@echo $(CONTAINER_TAG)

#----------------------------------------------------------------------
# CentOS 7 build API END
#----------------------------------------------------------------------

.PHONY: ALWAYS_REBUILD
.ALWAYS_REBUILD:
