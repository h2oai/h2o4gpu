location = $(CURDIR)/$(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))
WHERE := $(location)
$(info ** -> $(WHERE))
$(info ** ------------------------------------------------------------------ **)

NVCC := $(shell command -v nvcc 2> /dev/null)

#local settings
USENCCL=0
USENVTX=0
CMAKE_BUILD_TYPE=0

$(warning CMAKE_BUILD_TYPE is $(CMAKE_BUILD_TYPE))
$(warning USENVTX is $(USENVTX))
$(warning USENCCL is $(USENCCL))

# for R (rest can do both at same time)
#TARGET=gpulib
#$(warning R TARGET is $(TARGET))

ifdef NVCC
# CUDA Flags for XGBoost
CUDA_LIB=$(CUDA_HOME)/lib64
CUDA_VERSION ?= $(shell ls $(CUDA_LIB)/libcudart.so.* | head -1 | rev | cut -d "." -f -2 | rev)
CUDA_MAJOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 1)
ifeq ($(shell test $(CUDA_MAJOR) -ge 9; echo $$?),0)
    $(warning Compiling with Cuda9 or higher)
    # >=52 required for kmeans for larger data of size rows/32>2^16
    XGB_CUDA ?= -DGPU_COMPUTE_VER="35;52;60;61;70"
else
    $(warning Compiling with Cuda8 or lower)
    # >=52 required for kmeans for larger data of size rows/32>2^16
    XGB_CUDA ?= -DGPU_COMPUTE_VER="35;52;60;61"
endif
else
$(warning No CUDA found.)
endif
