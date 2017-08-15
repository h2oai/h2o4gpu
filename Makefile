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
	@echo "make          Compile and Install everything"
	@echo "make all      Compile and Install everything"
	@echo "make allclean Clean everything, then Compile and Install everything"
	@echo "make test     Run tests"
	@echo "make clean    Clean all build files"
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

sync_data: sync_smalldata sync_otherdata

default: all

all: cpp c py r

cpp:
	$(MAKE) -j all -C src/
	$(MAKE) -j all -C examples/cpp/

c:
	$(MAKE) -j all -C src/interface_c

py:
	$(MAKE) -j all -C src/interface_py

r:
	$(MAKE) -j all -C src/interface_r


veryallclean: clean deps_fetch deps_install all

allclean: clean all

clean: cleancpp cleanc cleanpy cleanr deps_clean

cleancpp:
	$(MAKE) -j clean -C src/
	$(MAKE) -j clean -C examples/cpp/

cleanc:
	$(MAKE) -j clean -C src/interface_c

cleanpy:
	$(MAKE) -j clean -C src/interface_py

cleanr:
	$(MAKE) -j clean -C src/interface_r

getotherdata:
	cd ~/h2oai-prototypes/glm-bench/ ; gunzip -f ipums.csv.gz ; Rscript ipums_feather.R ; cd ~/h2ogpuml/testsbig/data/ ; ln -sf ~/h2oai-prototypes/glm-bench/ipums.feather .

dotest:
	mkdir -p ./tmp/
	pytest -s --verbose --durations=10 -n auto --fulltrace --full-trace --junit-xml=build/test-reports/h2oai-test.xml tests 2> ./tmp/h2oai-test.$(LOGEXT).log

dotestbig:
	mkdir -p ./tmp/
	pytest -s --verbose --durations=10 -n auto --fulltrace --full-trace --junit-xml=build/test-reports/h2oai-test.xml testsbig 2> ./tmp/h2oai-test.$(LOGEXT).log

test: all sync_data dotest

testbig: all sync_data dotestbig

testquick: dotest

testbigquick: dotestbig

deps_clean: 
	@echo "----- Cleaning deps -----"
	rm -rf "$(DEPS_DIR)"

deps_fetch: deps_clean
	@echo "---- Fetch dependencies ---- "
	@mkdir -p "$(DEPS_DIR)"
	$(S3_CMD_LINE) get "$(ARTIFACTS_BUCKET)/ai/h2o/pydatatable/$(PYDATATABLE_VERSION)/*.whl" "$(DEPS_DIR)/"
	@find "$(DEPS_DIR)" -name "*.whl" | grep -i $(PY_OS) > "$(DEPS_DIR)/requirements.txt"
	@echo "** Local Python dependencies list for $(OS) stored in $(DEPS_DIR)/requirements.txt"
	bash gitshallow_submodules.sh

deps_install: deps_fetch sync_data
	@echo "---- Install dependencies ----"
	pip install -r "$(DEPS_DIR)/requirements.txt" --upgrade
	pip install -r requirements.txt --upgrade

