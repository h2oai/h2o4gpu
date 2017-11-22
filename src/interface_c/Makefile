
include ../config2.mk

H2O4GPUROOT=../

GITHASH := $(shell git rev-parse --short HEAD 2> /dev/null)
HASH=-D_GITHASH_=\"$(GITHASH)\"


ifeq ($(USEICC),0) # assumes only gnu not pgi
#MKLSTUFF=-lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -liomp5 -lpthread
MKLSTUFF=-lmkl_def -lmkl_avx -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread
else
MKLSTUFF=-lmkl_def -lmkl_avx -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread
endif
ICCSTUFF=-qopenmp
#GCCSTUFF=-fopenmp -ggdb3 -lgomp
GCCSTUFF=-fopenmp -lgomp


# C++ Flags
ifeq ($(USEICC),0)
CXX=g++
ICCSTUFF=
else
CXX=icpc
GCCSTUFF=
endif

ifeq ($(USEMKL),0)
MKLSTUFF=-lm -lblas
endif

LINKFLAGS=$(ICCSTUFF) $(MKLSTUFF) $(GCCSTUFF)
ifeq ($(USENVTX),1)
LINKFLAGS+=-lnvToolsExt -L/usr/local/cuda/lib64/
endif
ifeq ($(USENCCL),1)
LINKFLAGS+=-lnccl
endif

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

CUDFLAGS=$(DFLAGS) -DHAVECUDA -DCUDCHECKERR


# 3. Set CUDA_HOME= path/to/CUDA/libraries/
ifndef CUDAHOME
ifeq ($(shell uname -s), Darwin)
CUDAHOME=/usr/local/cuda/lib/
else
CUDAHOME=/usr/local/cuda/lib64/
endif
endif

# Check System Args.
ifeq ($(shell uname -s), Darwin)
LDFLAGS=-lm -framework Accelerate
CULDFLAGS=-L$(CUDAHOME) -L/usr/local/lib $(CULDFLAGS_)
SHARED=dylib
else
LDFLAGS=$(LINKFLAGS)
CULDFLAGS=-L$(CUDAHOME) -lcudart -lcudadevrt -lcublas -lcusolver -lcusparse -lcurand $(LINKFLAGS) $(DFLAGS)
SHARED=so
endif

# C++ Flags
CXXFLAGS=$(IFLAGS) $(OPTFLAG) $(HASH) -I$(H2O4GPUROOT)/include -I$(CUDAHOME)/../include -std=c++11 -Wall -Wconversion -fPIC $(LINKFLAGS)  $(DFLAGS)
ifeq ($(USENVTX),1)
CXXFLAGS+=-I/usr/local/cuda/include/
endif

# H2O4GPU static libraries
H2O4GPULIB=\
	$(H2O4GPUROOT)/build/cpu/h2o4gpuglm.o \
	$(H2O4GPUROOT)/build/cpu/h2o4gpukmeans.o \
	$(H2O4GPUROOT)/build/cpu/matrix/matrix_dense.o \
	$(H2O4GPUROOT)/build/cpu/matrix/matrix_sparse.o \
	$(H2O4GPUROOT)/build/cpu/projector/projector_direct_dense.o \
	$(H2O4GPUROOT)/build/cpu/projector/projector_cgls.o \
	$(H2O4GPUROOT)/build/cpu/elastic_net_ptr.o

CUH2O4GPULIB=\
    $(H2O4GPUROOT)/build/gpu/h2o4gpu_link.o \
	$(H2O4GPUROOT)/build/gpu/h2o4gpuglm.o \
	$(H2O4GPUROOT)/build/gpu/kmeans/kmeans_h2o4gpu.o \
	$(H2O4GPUROOT)/build/gpu/kmeans/kmeans_labels.o \
	$(H2O4GPUROOT)/build/gpu/warmstart.o \
	$(H2O4GPUROOT)/build/gpu/utils.o \
	$(H2O4GPUROOT)/build/gpu/p2pbwcheck.o \
	$(H2O4GPUROOT)/build/gpu/bwcheck.o \
	$(H2O4GPUROOT)/build/gpu/matrix/matrix_dense.o \
	$(H2O4GPUROOT)/build/gpu/matrix/matrix_sparse.o \
	$(H2O4GPUROOT)/build/gpu/projector/projector_direct_dense.o \
	$(H2O4GPUROOT)/build/gpu/projector/projector_cgls.o \
	$(H2O4GPUROOT)/build/gpu/elastic_net_ptr.o \
	$(H2O4GPUROOT)/build/gpu/pca/pca.o   \
	$(H2O4GPUROOT)/build/gpu/tsvd/tsvd.o   \
    $(H2O4GPUROOT)/build/gpu/data/matrix.o \
	$(H2O4GPUROOT)/build/common/logger.o \
	$(H2O4GPUROOT)/build/common/utils.o

# Python-required CUDA runtime library
PYCUDART=$(CUDAHOME)/libcudart.so


.PHONY: default all clean cpulib gpulib

default: all

all: cpulib gpulib ch2o4gpu_cpu ch2o4gpu_gpu h2o4gpu_c.o

cpulib:
	$(MAKE) cpulib -C $(H2O4GPUROOT) IFLAGS=$(IFLAGS)

gpulib:
	$(MAKE) gpulib -C $(H2O4GPUROOT) IFLAGS=$(IFLAGS)

ch2o4gpu_cpu: h2o4gpu_c.o  Makefile
	touch ch2o4gpu_cpu
	$(MAKE) cpulib -C $(H2O4GPUROOT) IFLAGS=$(IFLAGS)
	$(CXX) $(CXXFLAGS) -shared -o $@.$(SHARED) $< $(H2O4GPULIB) $(LDFLAGS)

ch2o4gpu_gpu: h2o4gpu_c.o Makefile
	touch ch2o4gpu_gpu
	$(MAKE) gpulib -C $(H2O4GPUROOT) IFLAGS=$(IFLAGS)
	$(CXX) $(CUDFLAGS) -shared -o $@.$(SHARED) $< $(CUH2O4GPULIB) $(PYCUDART) $(CULDFLAGS)

h2o4gpu_c.o: h2o4gpu_c.cpp h2o4gpu_c.h Makefile
	$(CXX) $(CXXFLAGS) $< -c -o $@

clean:
	rm -fv *.o *.so *.dylib
	rm -rf ch2o4gpu_cpu ch2o4gpu_gpu


