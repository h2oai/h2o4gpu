include ../config2.mk

H2OAIGLMROOT=../

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
ifeq ($(USEDEBUG),1)
DFLAGS+=-DDEBUG
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
CXXFLAGS=$(IFLAGS) -O3 $(HASH) -I$(H2OAIGLMROOT)/include -I$(CUDAHOME)/../include -std=c++11 -Wall -Wconversion -fPIC $(LINKFLAGS)  $(DFLAGS)
ifeq ($(USENVTX),1)
CXXFLAGS+=-I/usr/local/cuda/include/
endif

# H2OAIGLM static libraries
H2OAIGLMLIB=\
	$(H2OAIGLMROOT)/build/cpu/h2oaiglm.o \
	$(H2OAIGLMROOT)/build/cpu/matrix/matrix_dense.o \
	$(H2OAIGLMROOT)/build/cpu/matrix/matrix_sparse.o \
	$(H2OAIGLMROOT)/build/cpu/projector/projector_direct_dense.o \
	$(H2OAIGLMROOT)/build/cpu/projector/projector_cgls.o \
	$(H2OAIGLMROOT)/build/cpu/elastic_net_ptr.o

CUH2OAIGLMLIB=\
    $(H2OAIGLMROOT)/build/gpu/h2oaiglm_link.o \
	$(H2OAIGLMROOT)/build/gpu/h2oaiglm.o \
	$(H2OAIGLMROOT)/build/gpu/h2oaikmeans.o \
	$(H2OAIGLMROOT)/build/gpu/kmeans_labels.o \
	$(H2OAIGLMROOT)/build/gpu/warmstart.o \
	$(H2OAIGLMROOT)/build/gpu/p2pbwcheck.o \
	$(H2OAIGLMROOT)/build/gpu/bwcheck.o \
	$(H2OAIGLMROOT)/build/gpu/matrix/matrix_dense.o \
	$(H2OAIGLMROOT)/build/gpu/matrix/matrix_sparse.o \
	$(H2OAIGLMROOT)/build/gpu/projector/projector_direct_dense.o \
	$(H2OAIGLMROOT)/build/gpu/projector/projector_cgls.o \
	$(H2OAIGLMROOT)/build/gpu/elastic_net_ptr.o

# Python-required CUDA runtime library
PYCUDART=$(CUDAHOME)/libcudart.so	


.PHONY: default all clean cpulib gpulib

default: all

all: cpulib gpulib ch2oaiglm_cpu ch2oaiglm_gpu h2oaiglm_c.o

cpulib:
	$(MAKE) cpulib -C $(H2OAIGLMROOT) IFLAGS=$(IFLAGS)

gpulib:
	$(MAKE) gpulib -C $(H2OAIGLMROOT) IFLAGS=$(IFLAGS)

ch2oaiglm_cpu: h2oaiglm_c.o  Makefile
	touch ch2oaiglm_cpu
	$(MAKE) cpulib -C $(H2OAIGLMROOT) IFLAGS=$(IFLAGS)
	$(CXX) $(CXXFLAGS) -shared -o $@.$(SHARED) $< $(H2OAIGLMLIB) $(LDFLAGS)

ch2oaiglm_gpu: h2oaiglm_c.o Makefile
	touch ch2oaiglm_gpu
	$(MAKE) gpulib -C $(H2OAIGLMROOT) IFLAGS=$(IFLAGS)
	$(CXX) $(CUDFLAGS) -shared -o $@.$(SHARED) $< $(CUH2OAIGLMLIB) $(PYCUDART) $(CULDFLAGS)

h2oaiglm_c.o: h2oaiglm_c.cpp h2oaiglm_c.h Makefile
	$(CXX) $(CXXFLAGS) $< -c -o $@

clean:
	rm -fv *.o *.so *.dylib


