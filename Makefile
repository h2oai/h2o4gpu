# TODO(jon): ensure CPU-only can compile (i.e. no nvcc, etc.)
#
# Build specific config
#
CONFIG=make/config.mk
include $(CONFIG)

VERSION=make/version.mk
include $(VERSION)

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
S3_CMD_LINE := s3cmd --skip-existing
ifeq ($(shell test -n "$(AWS_ACCESS_KEY_ID)" -a -n "$(AWS_SECRET_ACCESS_KEY)" && printf "true"), true)
		S3_CMD_LINE += --access_key=$(AWS_ACCESS_KEY_ID) --secret_key=$(AWS_SECRET_ACCESS_KEY)
endif


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

sync_smalldata:
	@echo "---- Synchronizing test data ----"
	mkdir -p $(DATA_DIR)
	$(S3_CMD_LINE) sync --no-preserve "$(SMALLDATA_BUCKET)" "$(DATA_DIR)"

sync_otherdata:
	@echo "---- Synchronizing data dir in test/ ----"
	mkdir -p $(DATA_DIR)
	$(S3_CMD_LINE) sync --recursive --no-preserve "$(DATA_BUCKET)" "$(DATA_DIR)"

default: fullinstall

#########################################

update_submodule:
	echo ADD UPDATE SUBMODULE HERE

cpp:
	$(MAKE) -j all -C src/
	$(MAKE) -j all -C examples/cpp/

c:
	$(MAKE) -j all -C src/interface_c

py:
	$(MAKE) -j all -C src/interface_py

r:
	$(MAKE) -j all -C src/interface_r

pyinstall: py
	$(MAKE) -j install -C src/interface_py

rinstall: r
	$(MAKE) -j install -C src/interface_r

##############################################

alldeps: deps_fetch alldeps_install

build: update_submodule cpp c py r

install: pyinstall rinstall

fullinstall: clean alldeps build sync_smalldata install

#############################################

clean: cleancpp cleanc cleanpy cleanr deps_clean xgboost_clean py3nvml_clean
	rm -rf ./results/ ./tmp/

cleancpp:
	$(MAKE) -j clean -C src/
	$(MAKE) -j clean -C examples/cpp/

cleanc:
	$(MAKE) -j clean -C src/interface_c

cleanpy:
	$(MAKE) -j clean -C src/interface_py

cleanr:
	$(MAKE) -j clean -C src/interface_r

# uses https://github.com/Azure/fast_retraining
testxgboost:
	sh testsxgboost/runtestxgboost.sh

################

deps_clean: 
	@echo "----- Cleaning deps -----"
	rm -rf "$(DEPS_DIR)"
	# sometimes --upgrade leaves extra packages around
	sed 's/==.*//g' requirements.txt > requirements_plain.txt
	-xargs -a requirements_plain.txt -n 1 -P $(NUMPROCS) pip uninstall -y
	rm -rf requirements_plain.txt

deps_fetch:
	@echo "---- Fetch dependencies ---- "
	@mkdir -p "$(DEPS_DIR)"
	$(S3_CMD_LINE) get "$(ARTIFACTS_BUCKET)/ai/h2o/pydatatable/$(PYDATATABLE_VERSION)/*.whl" "$(DEPS_DIR)/"
	@find "$(DEPS_DIR)" -name "*.whl" | grep -i $(PY_OS) > "$(DEPS_DIR)/requirements.txt"
	@echo "** Local Python dependencies list for $(OS) stored in $(DEPS_DIR)/requirements.txt"
	bash scripts/gitshallow_submodules.sh

deps_install:
	@echo "---- Install dependencies ----"
	#-xargs -a "$(DEPS_DIR)/requirements.txt" -n 1 -P 1 pip install --upgrade
	#-xargs -a requirements.txt -n 1 -P 1 pip install --upgrade
	pip install -r "$(DEPS_DIR)/requirements.txt" --upgrade --no-cache-dir
	pip install -r requirements.txt --upgrade --no-cache-dir

alldeps_install: deps_install libxgboost libpy3nvml

###################

wheel_in_docker:
	docker build -t opsh2oai/h2o4gpu-build -f Dockerfile-build .
	docker run --rm -u `id -u`:`id -g` -v `pwd`:/work -w /work --entrypoint /bin/bash opsh2oai/h2o4gpu-build -c '. /h2oai_env/bin/activate; make update_submodule cpp c py'

clean_in_docker:
	docker build -t opsh2oai/h2o4gpu-build -f Dockerfile-build .
	docker run --rm -u `id -u`:`id -g` -v `pwd`:/work -w /work --entrypoint /bin/bash opsh2oai/h2o4gpu-build -c '. /h2oai_env/bin/activate; make clean'

###################
xgboost_clean:
	-pip uninstall -y xgboost
	rm -rf xgboost/build/

libxgboost:
	cd xgboost && git submodule init && git submodule update dmlc-core && git submodule update nccl && git submodule update cub && git submodule update rabit && mkdir -p build && cd build && cmake .. -DPLUGIN_UPDATER_GPU=ON -DCMAKE_BUILD_TYPE=Release && make -j  && cd ../python-package ; rm -rf dist && python setup.py sdist bdist_wheel && cd dist && pip install xgboost-0.6-py3-none-any.whl --upgrade --root=.

py3nvml_clean:
	-pip uninstall -y py3nvml

libpy3nvml:
	cd py3nvml # ; pip install -e git+https://github.com/fbcotter/py3nvml#egg=py3nvml --upgrade --root=.


liblightgbm:
	rm -rf LightGBM ; result=`git clone --recursive https://github.com/Microsoft/LightGBM`
	cd LightGBM && mkdir build ; cd build && cmake .. -DUSE_GPU=1 -DOpenCL_LIBRARY=$(CUDA_HOME)/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=$(CUDA_HOME)/include/ && make -j && cd ../python-package ; python setup.py install --precompile && cd ../ && pip install arff tqdm keras runipy --upgrade

#################### Jenkins specific

cleanjenkins: cleancpp cleanc cleanpy cleanr xgboost_clean py3nvml_clean

buildjekins: update_submodule cpp c py # r -- not yet

installjenkins: pyinstall # rinstall -- not yet

fullinstalljenkins: cleanjenkins alldeps buildjekins installjenkins

.PHONY: mrproper
mrproper: clean
	@echo "----- Cleaning properly -----"
	git clean -f -d -x

#################### H2O.ai specific

fullinstallprivate: clean build alldeps sync_data install

sync_data: sync_smalldata sync_otherdata


##################

dotest:
	rm -rf ./tmp/
	rm -rf build/test-reports 2>/dev/null
	mkdir -p ./tmp/
	pytest -s --verbose --durations=10 -n auto --fulltrace --full-trace --junit-xml=build/test-reports/h2o4gpu-test.xml tests 2> ./tmp/h2o4gpu-testbig.$(LOGEXT).log

dotestbig:
	mkdir -p ./tmp/
	pytest -s --verbose --durations=10 -n 1 --fulltrace --full-trace --junit-xml=build/test-reports/h2o4gpu-testbig.xml testsbig 2> ./tmp/h2o4gpu-test.$(LOGEXT).log

#####################

dotestperf:
	mkdir -p ./tmp/
	H2OGLM_PERFORMANCE=1 pytest -s --verbose --durations=10 -n 1 --fulltrace --full-trace --junit-xml=build/test-reports/h2o4gpu-test.xml tests 2> ./tmp/h2o4gpu-test.$(LOGEXT).log
	bash tests/showresults.sh

dotestbigperf:
	mkdir -p ./tmp/
	H2OGLM_PERFORMANCE=1 pytest -s --verbose --durations=10 -n 1 --fulltrace --full-trace --junit-xml=build/test-reports/h2o4gpu-test.xml testsbig 2> ./tmp/h2o4gpu-test.$(LOGEXT).log
	bash tests/showresults.sh # still just references results directory in base path

#########################

dotestperfpython:
	mkdir -p ./tmp/
	bash tests/getresults.sh $(LOGEXT)
	bash tests/showresults.sh

dotestbigperfpython:
	mkdir -p ./tmp/
	bash testsbig/getresultsbig.sh $(LOGEXT)
	bash tests/showresults.sh # still just references results directory in base path

################### H2O.ai private tests for pass/fail

test: build sync_data dotest

testbig: build sync_data dotestbig

testquick: dotest

testbigquick: dotestbig

################ H2O.ai private tests for performance

testperf: build sync_data dotestperf

testbigperf: build sync_data dotestbigperf

testperfquick: dotestperf

testbigperfquick: dotestbigperf

#################### Build info

H2O4GPU_COMMIT := $(shell git rev-parse HEAD)

.PHONY: base_version
base_version:
	@echo $(BASE_VERSION)

.buildinfo:
	@rm -rf $@
	@mkdir -p $@

# Generate local build info
.buildinfo/BUILD_INFO.txt: | .buildinfo
	@echo "build=\"LOCAL DEV (`git rev-parse --short HEAD` build at `date`)\"" > $@
	@echo "branch=\"`git rev-parse HEAD | git branch -a --contains | grep -v detached | sed -e 's~remotes/origin/~~g' -e 's~^ *~~' | sort | uniq | tr '*\n' ' '`\"" >> $@
	@echo "describe=\"`git describe --always --dirty`\"" >> $@
	@echo "build_os=\"`uname -a`\"" >> $@
	@echo "build_machine=\"`hostname`\"" >> $@
	@echo "build_date=\"`date`\"" >> $@
	@echo "build_user=\"`id -u -n`\"" >> $@
	@echo "base_version=\"$(BASE_VERSION)\"" >> $@
	@echo "h2oai_commit=\"$(H2O4GPU_COMMIT)\"" >> $@

src/interface_py/h2o4gpu/BUILD_INFO.txt: .buildinfo/BUILD_INFO.txt
	@rm -rf $@
	cp .buildinfo/BUILD_INFO.txt $@

build/VERSION.txt: src/interface_py/h2o4gpu/BUILD_INFO.txt
	@rm -rf build
	@mkdir -p build
	cd src/interface_py/; python setup.py --version > ../../build/VERSION.txt 2>/dev/null

# Refresh the build info only locally, let Jenkins to generate its own
ifeq ($(CI),)
.buildinfo/BUILD_INFO.txt: .ALWAYS_REBUILD
endif

.PHONY: ALWAYS_REBUILD
.ALWAYS_REBUILD:
