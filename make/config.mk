#
# Build configuration
#

# Location of artifacts
# E.g. "s3://bucket/dirname"
ARTIFACTS_BUCKET = s3://artifacts.h2o.ai/releases
# Location of local directory with dependencies
DEPS_DIR = deps

# Location of datasets
SMALLDATA_BUCKET = s3://h2o-public-test-data/smalldata

# Location of other datasets that we add for h2o4gpu testing (ipums, bnp, etc)
DATA_BUCKET = s3://h2o-datasets/h2o4gpu/data

# Location of local directory with data
DATA_DIR = data

# Location of open data
OPEN_DATA_BUCKET = s3://h2o-public-test-data/h2o4gpu/open_data

# Location of local directory with open data
OPEN_DATA_DIR = open_data

#
# PyDataTable
#
#PYDATATABLE_VERSION = 0.1.0+master.97

#
# XGBoost
#
XGBOOST_VERSION = 0.6

#
# R package Configurations
#
INSTALL_R = 1
R_VERSION = 3.1.0

#
# Find NVML library

# Certain dockers have only stubs under: /usr/local/cuda-X.Y/targets/ARCH/lib/stubs/libnvidia-ml.so
# Certain OS's have it under /lib, /lib64, /usr/lib or /usr/lib64
NVML_LIB := $(shell find /lib/ /lib64 /usr/lib /usr/lib64 /usr/local/ | grep libnvidia-ml.so | sort -n | head -1)

# Fail if not found - the build will pass but will fail during runtime
ifndef NVML_LIB
$(error NVML_LIB couldn't be found on your system. Please make sure libnvidia-ml.so is in /lib or /usr (os subdirectories))
endif

$(warning Compiling with NVML_LIB=$(NVML_LIB))
