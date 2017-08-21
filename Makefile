# TODO(jon): ensure CPU-only can compile (i.e. no nvcc, etc.)
#
# Build specific config
#
CONFIG=make/config.mk
include $(CONFIG)

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
	@echo "Example Pycharm environment flags: PYTHONPATH=/home/jon/h2ogpuml/src/interface_py:/home/jon/h2ogpuml;PYTHONUNBUFFERED=1;LD_LIBRARY_PATH=/opt/clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04//lib/:/home/jon/lib:/opt/rstudio-1.0.136/bin/:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64::/home/jon/lib/:$LD_LIBRARY_PATH;LLVM4=/opt/clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04/"
	@echo "Example Pycharm working directory: /home/jon/h2ogpuml/"

sync_smalldata:
	@echo "---- Synchronizing test data ----"
	mkdir -p $(DATA_DIR)
	$(S3_CMD_LINE) sync --no-preserve "$(SMALLDATA_BUCKET)" "$(DATA_DIR)"

sync_otherdata:
	@echo "---- Synchronizing data dir in test/ ----"
	mkdir -p $(DATA_DIR)
	$(S3_CMD_LINE) sync --no-preserve "$(DATA_BUCKET)" "$(DATA_DIR)"

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

fullinstall: clean build alldeps sync_smalldata install

#############################################

clean: cleancpp cleanc cleanpy cleanr deps_clean xgboost_clean py3nvml_clean
	rm -rf ./results/

cleancpp:
	$(MAKE) -j clean -C src/
	$(MAKE) -j clean -C examples/cpp/

cleanc:
	$(MAKE) -j clean -C src/interface_c

cleanpy:
	$(MAKE) -j clean -C src/interface_py

cleanr:
	$(MAKE) -j clean -C src/interface_r


##################

getotherdata:
	cd ~/h2oai-prototypes/glm-bench/ ; gunzip -f ipums.csv.gz ; Rscript ipums_feather.R ; cd ~/h2ogpuml/testsbig/data/ ; ln -sf ~/h2oai-prototypes/glm-bench/ipums.feather .

##################

dotest:
	rm -rf ./tmp/
	rm -rf build/test-reports 2>/dev/null
	mkdir -p ./tmp/
	pytest -s --verbose --durations=10 -n auto --fulltrace --full-trace --junit-xml=build/test-reports/h2ogpuml-test.xml tests 2> ./tmp/h2ogpuml-testbig.$(LOGEXT).log

dotestbig:
	mkdir -p ./tmp/
	pytest -s --verbose --durations=10 -n 1 --fulltrace --full-trace --junit-xml=build/test-reports/h2ogpuml-testbig.xml testsbig 2> ./tmp/h2ogpuml-test.$(LOGEXT).log

#####################

dotestperf:
	mkdir -p ./tmp/
	H2OGLM_PERFORMANCE=1 pytest -s --verbose --durations=10 -n 1 --fulltrace --full-trace --junit-xml=build/test-reports/h2ogpuml-test.xml tests 2> ./tmp/h2ogpuml-test.$(LOGEXT).log
	bash showresults.sh

dotestbigperf:
	mkdir -p ./tmp/
	H2OGLM_PERFORMANCE=1 pytest -s --verbose --durations=10 -n 1 --fulltrace --full-trace --junit-xml=build/test-reports/h2ogpuml-test.xml testsbig 2> ./tmp/h2ogpuml-test.$(LOGEXT).log
	bash showresults.sh

#########################

dotestperfpython:
	mkdir -p ./tmp/
	bash getresults.sh $(LOGEXT)
	bash showresults.sh

dotestbigperfpython:
	mkdir -p ./tmp/
	bash getresultsbig.sh $(LOGEXT)
	bash showresults.sh

###################

test: build sync_data dotest

testbig: build sync_data dotestbig

testquick: dotest

testbigquick: dotestbig

################

testperf: build sync_data dotestperf

testbigperf: build sync_data dotestbigperf

testquickperf: dotestperf

testbigquickperf: dotestbigperf

# uses https://github.com/Azure/fast_retraining
testlightgdbvsxgb:
	export MOUNT_POINT=./data/

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
	bash gitshallow_submodules.sh

deps_install:
	@echo "---- Install dependencies ----"
	#-xargs -a "$(DEPS_DIR)/requirements.txt" -n 1 -P 1 pip install --upgrade
	#-xargs -a requirements.txt -n 1 -P 1 pip install --upgrade
	pip install -r "$(DEPS_DIR)/requirements.txt" --upgrade --no-cache-dir
	pip install -r requirements.txt --upgrade --no-cache-dir

alldeps_install: deps_install libxgboost libpy3nvml

###################

wheel_in_docker:
	docker build -t opsh2oai/h2ogpuml-build -f Dockerfile-build .
	docker run --rm -u `id -u`:`id -g` -v `pwd`:/work -w /work --entrypoint /bin/bash opsh2oai/h2ogpuml-build -c '. /h2oai_env/bin/activate; make update_submodule cpp c py'

clean_in_docker:
	docker build -t opsh2oai/h2ogpuml-build -f Dockerfile-build .
	docker run --rm -u `id -u`:`id -g` -v `pwd`:/work -w /work --entrypoint /bin/bash opsh2oai/h2ogpuml-build -c '. /h2oai_env/bin/activate; make clean'

###################
xgboost_clean:
	-pip uninstall -y xgboost
	rm -rf xgboost/build/

libxgboost:
	cd xgboost && git submodule init && git submodule update dmlc-core && git submodule update nccl && git submodule update cub && git submodule update rabit && mkdir -p build && cd build && cmake .. -DPLUGIN_UPDATER_GPU=ON -DCMAKE_BUILD_TYPE=Release && make -j  && cd ../python-package && python setup.py sdist bdist_wheel && cd dist && pip install xgboost-0.6-py3-none-any.whl --upgrade --root=.

py3nvml_clean:
	-pip uninstall -y py3nvml

libpy3nvml:
	cd py3nvml # ; pip install -e git+https://github.com/fbcotter/py3nvml#egg=py3nvml --upgrade --root=.


#################### Jenkins specific

cleanjenkins: cleancpp cleanc cleanpy cleanr xgboost_clean py3nvml_clean

buildjekins: update_submodule cpp c py # r -- not yet

installjenkins: pyinstall # rinstall -- not yet

fullinstalljenkins: cleanjenkins buildjekins alldeps installjenkins

.PHONY: mrproper
mrproper: clean
	@echo "----- Cleaning properly -----"
	git clean -f -d -x

#################### H2O.ai specific

fullinstallprivate: clean build alldeps sync_data sync_smalldata install

sync_data: sync_smalldata sync_otherdata

