SHELL := /bin/bash # force avoidance of dash as shell
include config2.mk

ifndef CUDA_HOME
$(error CUDA_HOME is not set)
endif

# Bulid directory
OBJDIR=build

RLIBHOME := $(shell R RHOME)
RINCLUDEHOME := $(shell R CMD config --cppflags)

GITHASH := $(shell git rev-parse --short HEAD 2> /dev/null)
HASH=-D_GITHASH_=\"$(GITHASH)\"

#https://software.intel.com/en-us/node/528685
ifeq ($(USEICC),0) # assumes only gnu not pgi
#MKLSTUFF=-lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -liomp5 -lpthread
MKLSTUFF=-lmkl_def -lmkl_avx -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread
else
MKLSTUFF=-lmkl_def -lmkl_avx -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread
endif
ICCSTUFF=-qopenmp
GCCSTUFF=-fopenmp #-ggdb3


# C++ Flags
ifeq ($(USEICC),0)
CXX=g++
ICCSTUFF=
else
CXX=icpc
GCCSTUFF=
endif

ifeq ($(USEMKL),0)
MKLSTUFF=
endif

# CUDA Flags
#CUDA_LIB=$(CUDA_HOME)/lib64
#CUDA_VERSION ?= $(shell ls $(CUDA_LIB)/libcudart.so.* | head -1 | rev | cut -d "." -f -2 | rev)
#CUDA_MAJOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 1)
#CUDA_MINOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 2)
CUB_HOME=../cub/

LINKFLAGS=$(ICCSTUFF) $(MKLSTUFF) $(GCCSTUFF)

DFLAGS=-DUSEICC=$(USEICC) -DUSEMKL=$(USEMKL)
OPTFLAG=-O3
ifeq ($(USEDEBUG),1)
DFLAGS+=-DDEBUG
OPTFLAG=-O0 -g # override default
endif
ifeq ($(USENCCL),1)
DFLAGS+=-DUSE_NCCL
endif
ifeq ($(USENVTX),1)
DFLAGS+=-DUSE_NVTX=1
endif

CUDFLAGS=$(DFLAGS) -DHAVECUDA -DCUDCHECKERR -DCUDA_MAJOR=$(CUDA_MAJOR) -DCUDA_MINOR=$(CUDA_MINOR)
ifeq ($(USENVTX),1)
CUDFLAGS+=-I/usr/local/cuda/include
endif

CXXFLAGS=$(IFLAGS) $(OPTFLAG) $(HASH) -Wall -std=c++11 -fPIC $(LINKFLAGS) $(RINCLUDEHOME) $(DFLAGS)
ifeq ($(USENVTX),1)
CXXFLAGS+=-I/usr/local/cuda/include
endif

ifeq ($(USEICC),0)
OPENMP=-fopenmp
NVCCEXTRA=
else
OPENMP=-qopenmp
NVCCEXTRA=-ccbin=icpc
endif

# Assume installed nccl into $(CUDA_HOME) and moved libraries in lib to lib64
CUXX=$(CUDA_HOME)/bin/nvcc --expt-relaxed-constexpr --expt-extended-lambda $(NVCCEXTRA) -L$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/include -I$(CUB_HOME) -lineinfo
CUFLAGS=$(IFLAGS) -std=c++11 $(OPTFLAG) $(HASH) -Xcompiler -fPIC -Xcompiler $(OPENMP) $(RINCLUDEHOME) $(CUDFLAGS)
CUFLAGS += -DCUDA_MAJOR=$(CUDA_MAJOR) -DCUDA_MINOR=$(CUDA_MINOR) -lcublas -lcusolver -lcusparse $(NVCC_GENCODE)
#-gencode arch=compute_20,code=sm_20

# Load local (e.g. R) specific config
include config.mk

# H2O4GPU header files.
H2O4GPU_HDR=\
	include/interface_defs.h \
	include/h2o4gpuglm.h \
	include/prox_lib.h \
	include/util.h \
	include/matrix/matrix.h \
	include/matrix/matrix_dense.h \
	include/matrix/matrix_sparse.h \
	include/projector/projector_cgls.h \
	include/projector/projector_direct.h

# CPU Specific headers and object files.
GSL_HDR=\
	cpu/include/gsl/cblas.h \
	cpu/include/gsl/gsl_blas.h \
	cpu/include/gsl/gsl_linalg.h \
	cpu/include/gsl/gsl_matrix.h \
	cpu/include/gsl/gsl_rand.h \
	cpu/include/gsl/gsl_spblas.h \
	cpu/include/gsl/gsl_spmat.h \
	cpu/include/gsl/gsl_vector.h

CPU_HDR=\
	cpu/include/cgls.h \
	cpu/include/equil_helper.h \
	cpu/include/projector_helper.h
CPU_MTX_OBJ=\
	$(OBJDIR)/cpu/matrix/matrix_sparse.o \
	$(OBJDIR)/cpu/matrix/matrix_dense.o
CPU_PRJ_OBJ=\
	$(OBJDIR)/cpu/projector/projector_cgls.o \
	$(OBJDIR)/cpu/projector/projector_direct_dense.o
CPU_OBJ=$(OBJDIR)/cpu/h2o4gpuglm.o $(OBJDIR)/cpu/elastic_net_ptr.o $(OBJDIR)/cpu/h2o4gpukmeans.o

H2O4GPUKMEANS_HDR=


# GPU Specific headers and object files.
CML_HDR=\
	gpu/include/cml/cblas.h \
	gpu/include/cml/cml_blas.cuh \
	gpu/include/cml/cml_defs.cuh \
	gpu/include/cml/cml_linalg.cuh \
	gpu/include/cml/cml_matrix.cuh \
	gpu/include/cml/cml_rand.cuh \
	gpu/include/cml/cml_spblas.cuh \
	gpu/include/cml/cml_spmat.cuh \
	gpu/include/cml/cml_utils.cuh \
	gpu/include/cml/cml_vector.cuh

GPU_HDR=\
	gpu/include/cgls.cuh \
	gpu/include/equil_helper.cuh \
	gpu/include/projector_helper.cuh
GPU_MTX_OBJ=\
	$(OBJDIR)/gpu/matrix/matrix_dense.o \
	$(OBJDIR)/gpu/matrix/matrix_sparse.o
GPU_PRJ_OBJ=\
	$(OBJDIR)/gpu/projector/projector_cgls.o \
	$(OBJDIR)/gpu/projector/projector_direct_dense.o

GPU_KMEANS_HDR=gpu/kmeans/kmeans_labels.h gpu/kmeans/kmeans_centroids.h gpu/kmeans/kmeans_impl.h common/logger.h
GPU_TSVD_HDR=gpu/tsvd/tsvd.h gpu/data/matrix.cuh gpu/tsvd/utils.cuh gpu/device/device_context.cuh
GPU_PCA_HDR=$(GPU_TSVD_HDR) gpu/pca/pca.h

GPU_OBJ=$(OBJDIR)/gpu/tsvd/tsvd.o \
        $(OBJDIR)/gpu/pca/pca.o \
        $(OBJDIR)/gpu/data/matrix.o \
	$(OBJDIR)/gpu/kmeans/kmeans_h2o4gpu.o \
	$(OBJDIR)/gpu/kmeans/kmeans_labels.o \
	$(OBJDIR)/gpu/h2o4gpuglm.o \
	$(OBJDIR)/gpu/warmstart.o  \
	$(OBJDIR)/gpu/utils.o  \
	$(OBJDIR)/gpu/p2pbwcheck.o  \
	$(OBJDIR)/gpu/bwcheck.o \
	$(OBJDIR)/gpu/elastic_net_ptr.o \
	$(OBJDIR)/common/logger.o \
	$(OBJDIR)/common/utils.o

# Set vpath for build
VPATH=cpu cpu/matrix cpu/projector gpu gpu/matrix gpu/projector

.PHONY: all default clean

all: gpulib cpulib

default: gpulib

# Build all
cpulib: $(CPU_OBJ) $(CPU_MTX_OBJ) $(CPU_PRJ_OBJ)
	touch cpulib
	ar cr $(OBJDIR)/cpu/h2o4gpu.a $^

gpulib: $(OBJDIR)/gpu/h2o4gpu_link.o $(GPU_OBJ) $(GPU_MTX_OBJ) $(GPU_PRJ_OBJ) $(CPU_PRJ_OBJ)
	touch gpulib
	ar cr $(OBJDIR)/gpu/h2o4gpu.a $^


# Directories
$(OBJDIR):
	mkdir -p $@

# Directories Common
$(OBJDIR)/common: $(OBJDIR)
	mkdir -p $@

# Directories CPU
$(OBJDIR)/cpu: $(OBJDIR)
	mkdir -p $@

$(OBJDIR)/cpu/matrix: $(OBJDIR)/cpu
	mkdir -p $@

$(OBJDIR)/cpu/projector: $(OBJDIR)/cpu
	mkdir -p $@

# Directories GPU
$(OBJDIR)/gpu: $(OBJDIR)
	mkdir -p $@

$(OBJDIR)/gpu/kmeans: $(OBJDIR)
	mkdir -p $@

$(OBJDIR)/gpu/pca: $(OBJDIR)
	mkdir -p $@

$(OBJDIR)/gpu/tsvd: $(OBJDIR)
	mkdir -p $@

$(OBJDIR)/gpu/data: $(OBJDIR)
	mkdir -p $@

$(OBJDIR)/gpu/matrix: $(OBJDIR)/gpu
	mkdir -p $@

$(OBJDIR)/gpu/projector: $(OBJDIR)/gpu
	mkdir -p $@



# H2O4GPU CPU objects
$(OBJDIR)/cpu/h2o4gpu.o: cpu/h2o4gpuglm.cpp $(H2O4GPU_HDR) $(GSL_HDR) | $(OBJDIR)/cpu Makefile config2.mk
	$(CXX) -Iinclude -Icpu/include $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/cpu/h2o4gpukmeans.o: cpu/h2o4gpukmeans.cpp $(H2O4GPUKMEANS_HDR) $(GSL_HDR) | $(OBJDIR)/cpu Makefile config2.mk
	$(CXX) -Iinclude -I/usr/local/cuda/include -Icpu/include $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/cpu/matrix/%.o: %.cpp $(CPU_HDR) | $(OBJDIR)/cpu/matrix Makefile config2.mk
	$(CXX) -Iinclude -Icpu/include $< $(CXXFLAGS) $(IFLAGS) -c -o $@

$(OBJDIR)/cpu/projector/%.o: %.cpp $(CPU_HDR) | $(OBJDIR)/cpu/projector Makefile config2.mk
	$(CXX) -Iinclude -Icpu/include $< $(CXXFLAGS) $(IFLAGS) -c -o $@

$(OBJDIR)/cpu/%.o: %.cpp $(CPU_HDR) | $(OBJDIR)/cpu Makefile config2.mk
	$(CXX) -Iinclude -Icpu/include $< $(CXXFLAGS) $(IFLAGS) -c -o $@

$(OBJDIR)/cpu/elastic_net_ptr.o: common/elastic_net_ptr.cpp common/elastic_net_ptr.h $(H2O4GPU_HDR) $(GSL_HDR) | $(OBJDIR)/common Makefile config2.mk
	$(CXX) -Iinclude -Icommon/include $< $(CXXFLAGS) -c -o $@


# H2O4GPU GPU objects
$(OBJDIR)/gpu/h2o4gpu_link.o: $(GPU_OBJ) $(GPU_MTX_OBJ) $(GPU_PRJ_OBJ) | $(OBJDIR) Makefile config2.mk
	$(CUXX) $(CUFLAGS) $^ -dlink -o $@

$(OBJDIR)/gpu/h2o4gpu.o: gpu/h2o4gpuglm.cu $(H2O4GPU_HDR) $(CML_HDR) $(GSL_HDR) | $(OBJDIR)/gpu Makefile config2.mk
	$(CUXX) -Iinclude -Igpu/include -Icpu/include $< $(CUFLAGS) -dc -o $@

$(OBJDIR)/gpu/warmstart.o: gpu/warmstart.cu $(H2O4GPU_HDR) $(CML_HDR) | $(OBJDIR)/gpu Makefile config2.mk
	$(CUXX) -Iinclude -Igpu/include $< $(CUFLAGS) -dc -o $@
$(OBJDIR)/gpu/p2pbwcheck.o: gpu/p2pbwcheck.cu $(H2O4GPU_HDR) $(CML_HDR) | $(OBJDIR)/gpu Makefile config2.mk
	$(CUXX) -Iinclude -Igpu/include $< $(CUFLAGS) -dc -o $@
$(OBJDIR)/gpu/bwcheck.o: gpu/bwcheck.cu $(H2O4GPU_HDR) $(CML_HDR) | $(OBJDIR)/gpu Makefile config2.mk
	$(CUXX) -Iinclude -Igpu/include $< $(CUFLAGS) -dc -o $@
$(OBJDIR)/gpu/utils.o: gpu/utils.cu $(H2O4GPU_HDR) $(CML_HDR) | $(OBJDIR)/gpu Makefile config2.mk
	$(CUXX) -Iinclude -Igpu/include $< $(CUFLAGS) -dc -o $@

$(OBJDIR)/gpu/matrix/%.o: %.cu $(GPU_HDR) | $(OBJDIR)/gpu/matrix Makefile config2.mk
	$(CUXX) -Iinclude -Igpu/include $< $(CUFLAGS) $(IFLAGS) -dc -o $@

$(OBJDIR)/gpu/projector/%.o: %.cu $(GPU_HDR) | $(OBJDIR)/gpu/projector Makefile config2.mk
	$(CUXX) -Iinclude -Igpu/include $< $(CUFLAGS) $(IFLAGS) -dc -o $@

$(OBJDIR)/gpu/%.o: %.cu $(GPU_HDR) | $(OBJDIR)/gpu Makefile config2.mk $(OBJDIR)/gpu/tsvd $(OBJDIR)/gpu/pca $(OBJDIR)/gpu/data
	$(CUXX) -Iinclude -Igpu/include $< $(CUFLAGS) $(IFLAGS) -dc -o $@

$(OBJDIR)/gpu/kmeans/*.o: gpu/kmeans/*.cu $(GPU_KMEANS_HDR) | $(OBJDIR)/gpu/kmeans
	$(CUXX) -Igpu/include $< $(CUFLAGS) $(IFLAGS) -dc -o $@

$(OBJDIR)/gpu/kmeans/kmeans_h2o4gpu.o: gpu/kmeans/kmeans_h2o4gpu.cu $(GPU_KMEANS_HDR) | $(OBJDIR)/gpu/kmeans Makefile config2.mk
	$(CUXX) -Iinclude -Igpu/include $< $(CUFLAGS) -dc -o $@

$(OBJDIR)/gpu/elastic_net_ptr.o: common/elastic_net_ptr.cpp common/elastic_net_ptr.h $(H2O4GPU_HDR) $(GSL_HDR) | $(OBJDIR)/common Makefile config2.mk
	$(CXX) $(CXXFLAGS) $(CUDFLAGS)  -L$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/include -Iinclude -Icommon/include $< $(CXXFLAGS) $(CUDFLAGS) -c -o $@

$(OBJDIR)/common/utils.o: common/utils.cpp common/utils.h $(GSL_HDR) | $(OBJDIR)/common Makefile config2.mk
	$(CXX) $(CXXFLAGS) -Iinclude -Icpu/include $< -c $(CXXFLAGS) -o $@


$(OBJDIR)/cpu/h2o4gpu.o: cpu/h2o4gpuglm.cpp $(H2O4GPU_HDR) $(GSL_HDR) | $(OBJDIR)/cpu Makefile config2.mk
	$(CXX) -Iinclude -Icpu/include $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/cpu/h2o4gpukmeans.o: cpu/h2o4gpukmeans.cpp $(H2O4GPUKMEANS_HDR) $(GSL_HDR) | $(OBJDIR)/cpu Makefile config2.mk
	$(CXX) -Iinclude -I/usr/local/cuda/include -Icpu/include $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/cpu/matrix/%.o: %.cpp $(CPU_HDR) | $(OBJDIR)/cpu/matrix Makefile config2.mk
	$(CXX) -Iinclude -Icpu/include $< $(CXXFLAGS) $(IFLAGS) -c -o $@

$(OBJDIR)/cpu/projector/%.o: %.cpp $(CPU_HDR) | $(OBJDIR)/cpu/projector Makefile config2.mk
	$(CXX) -Iinclude -Icpu/include $< $(CXXFLAGS) $(IFLAGS) -c -o $@

$(OBJDIR)/cpu/%.o: %.cpp $(CPU_HDR) | $(OBJDIR)/cpu Makefile config2.mk
	$(CXX) -Iinclude -Icpu/include $< $(CXXFLAGS) $(IFLAGS) -c -o $@


$(OBJDIR)/common/logger.o: common/logger.cpp common/logger.h | $(OBJDIR)/common Makefile config2.mk
	$(CXX) $(CXXFLAGS) $< -c $(CXXFLAGS) -o $@


clean:
	rm -rf h2o4gpu.a build/ *.dSYM
	find -name "*.so" -delete
	find -name "*.a" -delete
	find -name "*.o" -delete

include ../tests/cpp/Makefile

test_cpp: $(ALL_TEST)
	$(ALL_TEST)