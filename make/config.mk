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
#
ARCH       := $(shell getconf LONG_BIT)
OS         := $(shell cat /etc/issue)
RHEL_OS    := $(shell cat /etc/redhat-release)

# Gets Driver Branch
DRIVER_BRANCH := $(shell nvidia-smi | grep Driver | cut -f 3 -d' ' | cut -f 1 -d '.')

ifeq (${ARCH},$(filter ${ARCH},32 64))
    # If correct architecture and libnvidia-ml library is not found
    # within the environment, build using the stub library

    ifneq (,$(findstring Ubuntu,$(OS)))
        DEB := $(shell dpkg -l | grep cuda)
        ifneq (,$(findstring cuda, $(DEB)))
            NVML_LIB := /usr/lib/nvidia-$(DRIVER_BRANCH)
        else
            NVML_LIB := /lib${ARCH}
        endif
    endif

    ifneq (,$(findstring SUSE,$(OS)))
        RPM := $(shell rpm -qa cuda*)
        ifneq (,$(findstring cuda, $(RPM)))
            NVML_LIB := /usr/lib${ARCH}
        else
            NVML_LIB := /lib${ARCH}
        endif
    endif

    ifneq (,$(findstring CentOS,$(RHEL_OS)))
        RPM := $(shell rpm -qa cuda*)
        ifneq (,$(findstring cuda, $(RPM)))
            NVML_LIB := /usr/lib${ARCH}/nvidia
        else
            NVML_LIB := /lib${ARCH}
        endif
    endif

    ifneq (,$(findstring Red Hat,$(RHEL_OS)))
        RPM := $(shell rpm -qa cuda*)
        ifneq (,$(findstring cuda, $(RPM)))
            NVML_LIB := /usr/lib${ARCH}/nvidia
        else
            NVML_LIB := /lib${ARCH}
        endif
    endif

    ifneq (,$(findstring Fedora,$(RHEL_OS)))
        RPM := $(shell rpm -qa cuda*)
        ifneq (,$(findstring cuda, $(RPM)))
            NVML_LIB := /usr/lib${ARCH}/nvidia
        else
            NVML_LIB := /lib${ARCH}
        endif
    endif

else
    NVML_LIB := ../../lib${ARCH}/stubs/
    $(info "libnvidia-ml.so.1" not found, using stub library.)
endif

ifneq (${ARCH},$(filter ${ARCH},32 64))
	$(error Unknown architecture!)
endif

$(warning Compiling with ARCH=$(ARCH))
$(warning Compiling with OS=$(OS))
$(warning Compiling with RHEL_OS=$(RHEL_OS))
$(warning Compiling with NVML_LIB=$(NVML_LIB))
