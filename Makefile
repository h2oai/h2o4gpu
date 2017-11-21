
SHELL := /bin/bash # force avoidance of dash as shell
# TODO(jon): ensure CPU-only can compile (i.e. no nvcc, etc.)
#
# Build specific config
#
CONFIG=make/config.mk
include $(CONFIG)

VERSION=make/version.mk
include $(VERSION)

# System specific stuff
include src/config2.mk

ifeq ($(shell test $(CUDA_MAJOR) -ge 9; echo $$?),0)
$(warning Compiling with Cuda9 or higher)
XGB_CUDA ?= -DGPU_COMPUTE_VER="35;52;60;61;70"
else
$(warning Compiling with Cuda8 or lower)
# >=52 required for kmeans for larger data of size rows/32>2^16
XGB_CUDA ?= -DGPU_COMPUTE_VER="35;52;60;61"
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
	@echo "make                 fullinstall"
	@echo "make fullinstalldev  Clean everything, then compile and install project for development."
	@echo "make fullinstall     Clean everything, then compile and install everything."
	@echo "make clean           Clean all build files."
	@echo "make build           Build the whole project."
	@echo "make sync_smalldata  Syncs the data needed for tests."
	@echo "make test            Run tests."
	@echo "make testbig         Run tests for big data."
	@echo "make testperf        Run performance and accuracy tests."
	@echo "make testbigperf     Run performance and accuracy tests for big data."
	@echo "Example Pycharm environment flags: PYTHONPATH=/home/jon/h2o4gpu/src/interface_py:/home/jon/h2o4gpu;PYTHONUNBUFFERED=1;LD_LIBRARY_PATH=/opt/clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04//lib/:/home/jon/lib:/opt/rstudio-1.0.136/bin/:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64::/home/jon/lib/:$LD_LIBRARY_PATH;LLVM4=/opt/clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04/"
	@echo "Example Pycharm working directory: /home/jon/h2o4gpu/"
	@echo "make docker-build-nccl-cuda9 Build inside docker and save wheel to src/interface_py/dist?/"
	@echo "make docker-runtime-nccl-cuda9 Build runtime docker and save to local path"

sync_smalldata:
	@echo "---- Synchronizing test data ----"
	mkdir -p $(DATA_DIR)
	$(S3_CMD_LINE) sync --no-sign-request "$(SMALLDATA_BUCKET)" "$(DATA_DIR)"

sync_otherdata:
	@echo "---- Synchronizing data dir in test/ ----"
	mkdir -p $(DATA_DIR)
	$(S3_CMD_LINE) sync --recursive "$(DATA_BUCKET)" "$(DATA_DIR)"

sync_open_data:
	@echo "---- Synchronizing sklearn and other open data in home directory ----"
	mkdir -p $(OPEN_DATA_DIR)
	$(S3_CMD_LINE) sync --no-sign-request "$(OPEN_DATA_BUCKET)" "$(OPEN_DATA_DIR)"

default: fullinstall

#########################################

update_submodule:
	echo ADD UPDATE SUBMODULE HERE

cpp:
	$(MAKE) -j all -C src/
	$(MAKE) -j all -C examples/cpp/

c:
	$(MAKE) -j all -C src/interface_c

py: apply_sklearn_simple build/VERSION.txt
	$(MAKE) -j all -C src/interface_py

pylint:
	$(MAKE) pylint -C src/interface_py

fullpy: apply_sklearn_simple pylint

pyinstall:
	$(MAKE) -j install -C src/interface_py

##############################################

alldeps: deps_fetch alldeps_install
alldeps2: deps_fetch alldeps_install2

alldeps_private: deps_fetch private_deps_fetch private_deps_install alldeps_install
alldeps_private2: deps_fetch private_deps_fetch private_deps_install alldeps_install2

alldeps_private-nccl-cuda8: deps_fetch private_deps_fetch private_deps_install alldeps_install-nccl-cuda8
alldeps_private-nonccl-cuda8: deps_fetch private_deps_fetch private_deps_install alldeps_install-nonccl-cuda8
alldeps_private-nccl-cuda9: deps_fetch private_deps_fetch private_deps_install alldeps_install-nccl-cuda9
alldeps_private-nonccl-cuda9: deps_fetch private_deps_fetch private_deps_install alldeps_install-nonccl-cuda9


build: update_submodule cleanbuild cpp c py

buildnocpp: update_submodule cleanc cleanpy c py # avoid cpp

buildquick: cpp c py

install: pyinstall

fullbuild: clean alldeps sync_open_data build
	mkdir -p src/interface_py/dist1/ && cp -a src/interface_py/dist/*.whl src/interface_py/dist1/
fullbuild-nonccl: clean alldeps2 sync_open_data build
	mkdir -p src/interface_py/dist2/ && cp -a src/interface_py/dist/*.whl src/interface_py/dist2/


fullinstall: clean alldeps sync_open_data build install
	mkdir -p src/interface_py/dist1/ && cp -a src/interface_py/dist/*.whl src/interface_py/dist1/
fullinstall-nonccl: clean alldeps2 sync_open_data build install
	mkdir -p src/interface_py/dist2/ && cp -a src/interface_py/dist/*.whl src/interface_py/dist2/

####################################################
# Docker stuff

docker-build-nccl-cuda9:
	@echo "+-- Building Wheel in Docker (-nccl-cuda9) --+"
	export CONTAINER_NAME="localmake-build" ;\
	export versionTag="0.0.4" ;\
	export extratag="-nccl-cuda9" ;\
	export encodedFullVersionTag="0.0.4" ;\
	export fullVersionTag="0.0.4" ;\
	export buckettype="releases/bleeding-edge" ;\
	export dockerimage="nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04" ;\
	export H2O4GPU_BUILD="" ;\
	export H2O4GPU_SUFFIX="" ;\
	export makeopts="" ;\
	export dist="dist" ;\
	bash scripts/make-docker-devel.sh

docker-runtime-nccl-cuda9:
	@echo "+--Building Runtime Docker Image Part 2 (-nccl-cuda9) --+"
	export CONTAINER_NAME="localmake-runtime" ;\
	export versionTag="0.0.4" ;\
	export extratag="-nccl-cuda9" ;\
	export encodedFullVersionTag=0.0.4 ;\
	export fullVersionTag=0.0.4 ;\
	export buckettype="releases/bleeding-edge" ;\
	export dockerimage="nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04" ;\
	bash scripts/make-docker-runtime.sh

docker-runtime-nccl-cuda9-load:
	nvidia-docker load < h2o4gpu-0.0.4-nccl-cuda9-runtime.tar.gz

.PHONY: docker-runtime-nccl-cuda9-run

docker-runtime-nccl-cuda9-run:
	@echo "+-Running Docker Runtime Image (-nccl-cuda9) --+"
	export CONTAINER_NAME="localmake-runtime-run" ;\
	export versionTag="0.0.4" ;\
	export extratag="-nccl-cuda9" ;\
	export encodedFullVersionTag="0.0.4" ;\
	export fullVersionTag="0.0.4" ;\
	export buckettype="releases/bleeding-edge" ;\
	export dockerimage="nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04" ;\
	export H2O4GPU_BUILD="" ;\
	export H2O4GPU_SUFFIX="" ;\
	export makeopts="" ;\
	export dist="dist" ;\
	nvidia-docker run --init --rm --name $${CONTAINER_NAME} -d -t -u `id -u`:`id -g` --entrypoint=bash opsh2oai/h2o4gpu-$${versionTag}$${extratag}-runtime:latest

docker-runtime-nccl-cuda8:
	@echo "+--Building Runtime Docker Image Part 2 (-nccl-cuda8) --+"
	export CONTAINER_NAME="localmake-runtime" ;\
	export versionTag="0.0.4" ;\
	export extratag="-nccl-cuda8" ;\
	export encodedFullVersionTag=0.0.4 ;\
	export fullVersionTag=0.0.4 ;\
	export buckettype="releases/bleeding-edge" ;\
	export dockerimage="nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04" ;\
	bash scripts/make-docker-runtime.sh

docker-runtests-nccl-cuda9:
	@echo "+-- Run tests in docker (-nccl-cuda9) --+"
	export CONTAINER_NAME="localmake-runtests" ;\
	export versionTag="0.0.4" ;\
	export extratag="-nccl-cuda8" ;\
	export encodedFullVersionTag=0.0.4 ;\
	export fullVersionTag=0.0.4 ;\
	export buckettype="releases/bleeding-edge" ;\
	export dockerimage="nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04" ;\
	export H2O4GPU_BUILD="" ;\
	export H2O4GPU_SUFFIX="" ;\
	export target=fullinstalljenkins${extratag} ;\
	export dist="dist" ;\
	bash scripts/make-docker-runtests.sh

get_docker:
	wget https://s3.amazonaws.com/artifacts.h2o.ai/releases/bleeding-edge/ai/h2o/h2o4gpu/0.0.4-nccl-cuda8/h2o4gpu-0.0.4-nccl-cuda8-runtime.tar.gz
load_docker:
	nvidia-docker load < h2o4gpu-0.0.4-nccl-cuda8-runtime.tar.gz

run_in_docker-nccl-cuda9:
	mkdir -p /home/$$USER/log ; chmod a+rwx /home/$$USER/log
	@echo "+-Running Docker Runtime Image (-nccl-cuda9) --+"
	export CONTAINER_NAME="localmake-runtime-run" ;\
	export versionTag="0.0.4" ;\
	export extratag="-nccl-cuda9" ;\
	export encodedFullVersionTag="0.0.4" ;\
	export fullVersionTag="0.0.4" ;\
	export buckettype="releases/bleeding-edge" ;\
	export dockerimage="nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04" ;\
	export H2O4GPU_BUILD="" ;\
	export H2O4GPU_SUFFIX="" ;\
	export makeopts="" ;\
	export dist="dist" ;\
	nvidia-docker run \
	--rm \
	--name $${CONTAINER_NAME} \
	-p 8888:8888 \
	-p 8889:8889 \
	-u `id -u`:`id -g` \
	-v /home/$$USER/log:/log \
	opsh2oai/h2o4gpu-$${versionTag}$${extratag}-runtime:latest


wheel_in_docker:
	docker build -t opsh2oai/h2o4gpu-build -f Dockerfile-build .
	docker run --rm -u `id -u`:`id -g` -v `pwd`:/work -w /work --entrypoint /bin/bash opsh2oai/h2o4gpu-build -c '. /h2oai_env/bin/activate; make update_submodule cpp c py'

wheel_in_docker-cuda9:
	docker build -t opsh2oai/h2o4gpu-cuda9-build -f Dockerfile-cuda9-build .
	docker run --rm -u `id -u`:`id -g` -v `pwd`:/work -w /work --entrypoint /bin/bash opsh2oai/h2o4gpu-cuda9-build -c '. /h2oai_env/bin/activate; make update_submodule cpp c py'

clean_in_docker:
	docker build -t opsh2oai/h2o4gpu-build -f Dockerfile-build .
	docker run --rm -u `id -u`:`id -g` -v `pwd`:/work -w /work --entrypoint /bin/bash opsh2oai/h2o4gpu-build -c '. /h2oai_env/bin/activate; make clean'

clean_in_docker-cuda9:
	docker build -t opsh2oai/h2o4gpu-cuda9-build -f Dockerfile-cuda9-build .
	docker run --rm -u `id -u`:`id -g` -v `pwd`:/work -w /work --entrypoint /bin/bash opsh2oai/h2o4gpu-cuda9-build -c '. /h2oai_env/bin/activate; make clean'

###################

#############################################



clean: cleanbuild deps_clean xgboost_clean py3nvml_clean
	-rm -rf ./build
	-rm -rf ./results/ ./tmp/

cleanbuild: cleancpp cleanc cleanpy

cleancpp:
	$(MAKE) -j clean -C src/
	$(MAKE) -j clean -C examples/cpp/

cleanc:
	$(MAKE) -j clean -C src/interface_c

cleanpy:
	$(MAKE) -j clean -C src/interface_py

# uses https://github.com/Azure/fast_retraining
testxgboost: # liblightgbm (assumes one installs lightgdm yourself or run make liblightgbm)
	bash testsxgboost/runtestxgboost.sh
	bash testsxgboost/extracttestxgboost.sh
	bash tests_open/showresults.sh # same for all tests

################

deps_clean:
	@echo "----- Cleaning deps -----"
	rm -rf "$(DEPS_DIR)"
	# sometimes --upgrade leaves extra packages around
	cat requirements_buildonly.txt requirements_runtime.txt > requirements.txt
	sed 's/==.*//g' requirements.txt > requirements_plain.txt
	-xargs -a requirements_plain.txt -n 1 -P $(NUMPROCS) pip uninstall -y
	rm -rf requirements_plain.txt requirements.txt

deps_fetch:
	@echo "---- Fetch dependencies ---- "
	bash scripts/gitshallow_submodules.sh

private_deps_fetch:
	@echo "---- Fetch private dependencies ---- "
	#@mkdir -p "$(DEPS_DIR)"
	#$(S3_CMD_LINE) get "$(ARTIFACTS_BUCKET)/ai/h2o/pydatatable/$(PYDATATABLE_VERSION)/*.whl" "$(DEPS_DIR)/"
	#@find "$(DEPS_DIR)" -name "*.whl" | grep -i $(PY_OS) > "$(DEPS_DIR)/requirements.txt"
	#@echo "** Local Python dependencies list for $(OS) stored in $(DEPS_DIR)/requirements.txt"

deps_install:
	@echo "---- Install dependencies ----"
	#-xargs -a requirements.txt -n 1 -P 1 pip install --upgrade
	easy_install pip
	easy_install setuptools
	cat requirements_buildonly.txt requirements_runtime.txt > requirements.txt
	pip install -r requirements.txt --upgrade
	rm -rf requirements.txt
	# issue with their package, have to do this here (still fails sometimes, so remove)
#	pip install sphinxcontrib-osexample

private_deps_install:
	@echo "---- Install private dependencies ----"
	#-xargs -a "$(DEPS_DIR)/requirements.txt" -n 1 -P 1 pip install --upgrade
	#pip install -r "$(DEPS_DIR)/requirements.txt" --upgrade

alldeps_install: deps_install apply_xgboost apply_py3nvml libsklearn # lib for sklearn because don't want to fully apply yet
alldeps_install2: deps_install apply_xgboost2 apply_py3nvml libsklearn # lib for sklearn because don't want to fully apply yet

alldeps_install-nccl-cuda8: deps_install apply_xgboost-nccl-cuda8 apply_py3nvml libsklearn # lib for sklearn because don't want to fully apply yet
alldeps_install-nonccl-cuda8: deps_install apply_xgboost-nonccl-cuda8 apply_py3nvml libsklearn # lib for sklearn because don't want to fully apply yet
alldeps_install-nccl-cuda9: deps_install apply_xgboost-nccl-cuda9 apply_py3nvml libsklearn # lib for sklearn because don't want to fully apply yet
alldeps_install-nonccl-cuda9: deps_install apply_xgboost-nonccl-cuda9 apply_py3nvml libsklearn # lib for sklearn because don't want to fully apply yet


xgboost_clean:
	-pip uninstall -y xgboost
	rm -rf xgboost/build/

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

# https://xgboost.readthedocs.io/en/latest/build.html
libxgboost: libxgboostp1 libxgboostp2 libxgboostp3
libxgboost2: libxgboostp1 libxgboostp2nonccl libxgboostp3

libxgboostp1:
	cd xgboost && git submodule init && git submodule update dmlc-core && git submodule update nccl && git submodule update cub && git submodule update rabit
libxgboostp2:
	cd xgboost && mkdir -p build && cd build && cmake .. -DUSE_CUDA=ON -DUSE_NCCL=ON $(XGB_CUDA) -DCMAKE_BUILD_TYPE=Release && make -j
libxgboostp2nonccl:
	cd xgboost && mkdir -p build && cd build && cmake .. -DUSE_CUDA=ON $(XGB_CUDA) -DCMAKE_BUILD_TYPE=Release && make -j
libxgboostp3:
	cd xgboost/python-package ; rm -rf dist && python setup.py sdist bdist_wheel

apply_xgboost: libxgboost pipxgboost
apply_xgboost2: libxgboost2 pipxgboost

apply_xgboost-nccl-cuda8: pipxgboost-nccl-cuda8
apply_xgboost-nonccl-cuda8:  pipxgboost-nonccl-cuda8
apply_xgboost-nccl-cuda9:  pipxgboost-nccl-cuda9
apply_xgboost-nonccl-cuda9:  pipxgboost-nonccl-cuda9


pipxgboost:
	@echo "----- pip install xgboost built locally -----"
	cd xgboost/python-package/dist && pip install xgboost-0.6-py3-none-any.whl --upgrade --target ../
	cd xgboost/python-package/xgboost ; cp -a ../lib/libxgboost*.so .

pipxgboost-nccl-cuda8:
	@echo "----- pip install xgboost-nccl-cuda8 from S3 -----"
	mkdir -p xgboost/python-package/dist ; cd xgboost/python-package/dist && pip install https://s3.amazonaws.com/artifacts.h2o.ai/releases/bleeding-edge/ai/h2o/xgboost/0.6-nccl-cuda8/xgboost-0.6-py3-none-any.whl --upgrade --target ../
pipxgboost-nonccl-cuda8:
	@echo "----- pip install xgboost-nonccl-cuda8 from S3 -----"
	mkdir -p xgboost/python-package/dist ; cd xgboost/python-package/dist && pip install https://s3.amazonaws.com/artifacts.h2o.ai/releases/bleeding-edge/ai/h2o/xgboost/0.6-nonccl-cuda8/xgboost-0.6-py3-none-any.whl --upgrade --target ../
pipxgboost-nccl-cuda9:
	@echo "----- pip install xgboost-nccl-cuda9 from S3 -----"
	mkdir -p xgboost/python-package/dist ; cd xgboost/python-package/dist && pip install https://s3.amazonaws.com/artifacts.h2o.ai/releases/bleeding-edge/ai/h2o/xgboost/0.6-nccl-cuda9/xgboost-0.6-py3-none-any.whl --upgrade --target ../
pipxgboost-nonccl-cuda9:
	@echo "----- pip install xgboost-nonccl-cuda9 from S3 -----"
	mkdir -p xgboost/python-package/dist ; cd xgboost/python-package/dist && pip install https://s3.amazonaws.com/artifacts.h2o.ai/releases/bleeding-edge/ai/h2o/xgboost/0.6-nonccl-cuda9/xgboost-0.6-py3-none-any.whl --upgrade --target ../

py3nvml_clean:
	-pip uninstall -y py3nvml

apply_py3nvml:
	cd py3nvml # ; pip install -e git+https://github.com/fbcotter/py3nvml#egg=py3nvml --upgrade --root=.


liblightgbm: # only done if user directly requests, never an explicit dependency
	echo "See https://github.com/Microsoft/LightGBM/wiki/Installation-Guide#with-gpu-support for details"
	echo "sudo apt-get install libboost-dev libboost-system-dev libboost-filesystem-dev cmake"
	rm -rf LightGBM ; result=`git clone --recursive https://github.com/Microsoft/LightGBM`
	cd LightGBM && mkdir build ; cd build && cmake .. -DUSE_GPU=1 -DOpenCL_LIBRARY=$(CUDA_HOME)/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=$(CUDA_HOME)/include/ && make -j && cd ../python-package ; python setup.py install --precompile --gpu && cd ../ && pip install arff tqdm keras runipy h5py --upgrade

libsklearn:	# assume already submodule gets sklearn
	@echo "----- Make sklearn wheel -----"
	bash scripts/prepare_sklearn.sh # repeated calls don't hurt
	rm -rf sklearn && mkdir -p sklearn && cd scikit-learn && python setup.py sdist bdist_wheel

apply_sklearn: libsklearn apply_sklearn_simple

apply_sklearn_simple:
    #	bash ./scripts/apply_sklearn.sh
    ## apply sklearn
	bash ./scripts/apply_sklearn_pipinstall.sh
    ## link-up recursively
	bash ./scripts/apply_sklearn_link.sh
    # handle base __init__.py file appending
	bash ./scripts/apply_sklearn_initmerge.sh

apply_sklearn_pipinstall:
	bash ./scripts/apply_sklearn_pipinstall.sh

apply_sklearn_link:
	bash ./scripts/apply_sklearn_link.sh

apply_sklearn_initmerge:
	bash ./scripts/apply_sklearn_initmerge.sh

#################### Jenkins specific

cleanjenkins: mrproper cleancpp cleanc cleanpy xgboost_clean py3nvml_clean

buildjenkins: update_submodule cpp c py

installjenkins: pyinstall

# for nccl cuda8 build
fullinstalljenkins-nccl-cuda8: cleanjenkins alldeps_private-nccl-cuda8 buildjenkins installjenkins
# for nonccl cuda8 build
fullinstalljenkins-nonccl-cuda8: cleanjenkins alldeps_private-nonccl-cuda8 buildjenkins installjenkins
	mkdir -p src/interface_py/dist2/ && mv src/interface_py/dist/*.whl src/interface_py/dist2/
# for nccl cuda9 build
fullinstalljenkins-nccl-cuda9: cleanjenkins alldeps_private-nccl-cuda9 buildjenkins installjenkins
	mkdir -p src/interface_py/dist4/ && mv src/interface_py/dist/*.whl src/interface_py/dist4/
# for nonccl cuda9 build
fullinstalljenkins-nonccl-cuda9: cleanjenkins alldeps_private-nonccl-cuda9 buildjenkins installjenkins
	mkdir -p src/interface_py/dist3/ && mv src/interface_py/dist/*.whl src/interface_py/dist3/

.PHONY: mrproper
mrproper: clean
	@echo "----- Cleaning properly -----"
	git clean -f -d -x

#################### H2O.ai specific

fullinstallprivate: clean alldeps_private build sync_data install
fullinstallprivate2: clean alldeps_private2 build sync_data install

sync_data: sync_otherdata sync_open_data # sync_smalldata  # not currently using smalldata

##################

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

dotestfast:
	rm -rf ./tmp/
	mkdir -p ./tmp/
    # can't do -n auto due to limits on GPU memory
	pytest -s --verbose --durations=10 -n 3 --fulltrace --full-trace --junit-xml=build/test-reports/h2o4gpu-testfast1.xml tests_open/test_glm_simple.py 2> ./tmp/h2o4gpu-testfast1.$(LOGEXT).log
	pytest -s --verbose --durations=10 -n 3 --fulltrace --full-trace --junit-xml=build/test-reports/h2o4gpu-testfast2.xml tests_open/test_xgb_sklearn_wrapper.py 2> ./tmp/h2o4gpu-testfast2.$(LOGEXT).log
	pytest -s --verbose --durations=10 -n 3 --fulltrace --full-trace --junit-xml=build/test-reports/h2o4gpu-testfast3.xml tests_open/test_tsvd.py 2> ./tmp/h2o4gpu-testfast3.$(LOGEXT).log
	pytest -s --verbose --durations=10 -n 3 --fulltrace --full-trace --junit-xml=build/test-reports/h2o4gpu-testfast4.xml tests_open/test_kmeans.py 2> ./tmp/h2o4gpu-testfast4.$(LOGEXT).log

dotestfast_nonccl:
	rm -rf ./tmp/
	mkdir -p ./tmp/
	# can't do -n auto due to limits on GPU memory
	pytest -s --verbose --durations=10 -n 3 --fulltrace --full-trace --junit-xml=build/test-reports/h2o4gpu-testfast1.xml tests_open/test_glm_simple.py 2> ./tmp/h2o4gpu-testfast1.$(LOGEXT).log
	pytest -s --verbose --durations=10 -n 3 --fulltrace --full-trace --junit-xml=build/test-reports/h2o4gpu-testfast3.xml tests_open/test_tsvd.py 2> ./tmp/h2o4gpu-testfast3.$(LOGEXT).log
	pytest -s --verbose --durations=10 -n 3 --fulltrace --full-trace --junit-xml=build/test-reports/h2o4gpu-testfast4.xml tests_open/test_kmeans.py 2> ./tmp/h2o4gpu-testfast4.$(LOGEXT).log

dotestsmall:
	rm -rf ./tmp/
	rm -rf build/test-reports 2>/dev/null
	mkdir -p ./tmp/
    # can't do -n auto due to limits on GPU memory
	pytest -s --verbose --durations=10 -n 4 --fulltrace --full-trace --junit-xml=build/test-reports/h2o4gpu-testsmall.xml tests_small 2> ./tmp/h2o4gpu-test.$(LOGEXT).log

dotestbig:
	mkdir -p ./tmp/
	pytest -s --verbose --durations=10 -n 1 --fulltrace --full-trace --junit-xml=build/test-reports/h2o4gpu-testbig.xml tests_big 2> ./tmp/h2o4gpu-test.$(LOGEXT).log

#####################

dotestperf:
	mkdir -p ./tmp/
	H2OGLM_PERFORMANCE=1 pytest -s --verbose --durations=10 -n 1 --fulltrace --full-trace --junit-xml=build/test-reports/h2o4gpu-test.xml tests_open 2> ./tmp/h2o4gpu-test.$(LOGEXT).log
	bash tests_open/showresults.sh

dotestsmallperf:
	mkdir -p ./tmp/
	H2OGLM_PERFORMANCE=1 pytest -s --verbose --durations=10 -n 1 --fulltrace --full-trace --junit-xml=build/test-reports/h2o4gpu-testsmallperf.xml tests_small 2> ./tmp/h2o4gpu-testperf.$(LOGEXT).log
	bash tests_open/showresults.sh

dotestbigperf:
	mkdir -p ./tmp/
	H2OGLM_PERFORMANCE=1 pytest -s --verbose --durations=10 -n 1 --fulltrace --full-trace --junit-xml=build/test-reports/h2o4gpu-testbigperf.xml tests_big 2> ./tmp/h2o4gpu-testbig.$(LOGEXT).log
	bash tests_open/showresults.sh # still just references results directory in base path

######################### use python instead of pytest (required in some cases if pytest leads to hang)

dotestperfpython:
	mkdir -p ./tmp/
	bash tests_open/getresults.sh $(LOGEXT)
	bash tests_open/showresults.sh

dotestbigperfpython:
	mkdir -p ./tmp/
	bash testsbig/getresultsbig.sh $(LOGEXT)
	bash tests_open/showresults.sh # still just references results directory in base path

################### H2O.ai public tests for pass/fail

testdemos: dotestdemos

test: build dotest # faster if also run sync_open_data before doing this test, but can't always assume user has s3 creds setup (even needed for public repo on S3)

testquick: dotest

################ H2O.ai public tests for performance

testperf: build dotestperf # faster if also run sync_open_data before doing this test

################### H2O.ai private tests for pass/fail

testsmall: build sync_data dotestsmall

testsmallquick: dotestsmall

testbig: build sync_data dotestbig

testbigquick: dotestbig

################ H2O.ai private tests for performance

testsmallperf: build sync_data dotestsmallperf

testbigperf: build sync_data dotestbigperf

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
	@echo "h2o4gpu_commit=\"$(H2OAI_COMMIT)\"" >> $@

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

.PHONY: ALWAYS_REBUILD
.ALWAYS_REBUILD:

