location = $(CURDIR)/$(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))
WHERE := $(location)
$(info ** -> $(WHERE))
$(info ** ------------------------------------------------------------------ **)

NVCC := $(shell command -v nvcc 2> /dev/null)

# For GPU case, must modify /usr/local/cuda/include/host_config.h to add && __ICC != 1700 to #error about unsupported ICC configuration
ICCFILE := $(shell command -v icpc 2> /dev/null)

ifdef ICCFILE
USEICC=1
else
USEICC=0
endif

ifdef MKLROOT
USEMKL=1
else
USEMKL=0
endif


#local settings
USEDEBUG=0
USENVTX=0
USENCCL=0


$(warning USEICC is $(USEICC))
$(warning ICCFILE is $(ICCFILE))
$(warning USEMKL is $(USEMKL))
$(warning USEDEBUG is $(USEDEBUG))
$(warning USENVTX is $(USENVTX))
$(warning USENCCL is $(USENCCL))


# for R (rest can do both at same time)
#TARGET=gpulib
#$(warning R TARGET is $(TARGET))

ifdef NVCC
# CUDA Flags
CUDA_LIB=$(CUDA_HOME)/lib64
CUDA_VERSION ?= $(shell ls $(CUDA_LIB)/libcudart.so.* | head -1 | rev | cut -d "." -f -2 | rev)
CUDA_MAJOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 1)
CUDA_MINOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 2)
ifeq ($(shell test $(CUDA_MAJOR) -ge 9; echo $$?),0)
$(warning Compiling with Cuda9 or higher)
# >=52 required for kmeans for larger data of size rows/32>2^16
NVCC_GENCODE ?= -gencode=arch=compute_35,code=sm_35 \
                -gencode=arch=compute_35,code=compute_35 \
				-gencode=arch=compute_52,code=sm_52 \
                -gencode=arch=compute_52,code=compute_52 \
                -gencode=arch=compute_60,code=sm_60 \
                -gencode=arch=compute_60,code=compute_60 \
                -gencode=arch=compute_61,code=sm_61 \
                -gencode=arch=compute_61,code=compute_61 \
                -gencode=arch=compute_70,code=sm_70 \
                -gencode=arch=compute_70,code=compute_70
XGB_CUDA ?= -DGPU_COMPUTE_VER="35;52;60;61;70"
else
$(warning Compiling with Cuda8 or lower)
# >=52 required for kmeans for larger data of size rows/32>2^16
NVCC_GENCODE ?= -gencode=arch=compute_35,code=sm_35 \
                -gencode=arch=compute_35,code=compute_35 \
				-gencode=arch=compute_52,code=sm_52 \
                -gencode=arch=compute_52,code=compute_52 \
                -gencode=arch=compute_60,code=sm_60 \
                -gencode=arch=compute_60,code=compute_60
                -gencode=arch=compute_61,code=sm_61 \
                -gencode=arch=compute_61,code=compute_61
XGB_CUDA ?= -DGPU_COMPUTE_VER="35;52;60;61"
endif
else
$(warning No CUDA found.)
endif
