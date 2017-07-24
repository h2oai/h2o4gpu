include ../config2.mk

H2OGPUMLROOT=../

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
CXXFLAGS=$(IFLAGS) -O3 $(HASH) -I$(H2OGPUMLROOT)/include -I$(CUDAHOME)/../include -std=c++11 -Wall -Wconversion -fPIC $(LINKFLAGS)  $(DFLAGS)
ifeq ($(USENVTX),1)
CXXFLAGS+=-I/usr/local/cuda/include/
endif

# H2OGPUML static libraries
H2OGPUMLLIB=\
	$(H2OGPUMLROOT)/build/cpu/h2ogpumlglm.o \
	$(H2OGPUMLROOT)/build/cpu/h2ogpumlkmeans.o \
	$(H2OGPUMLROOT)/build/cpu/matrix/matrix_dense.o \
	$(H2OGPUMLROOT)/build/cpu/matrix/matrix_sparse.o \
	$(H2OGPUMLROOT)/build/cpu/projector/projector_direct_dense.o \
	$(H2OGPUMLROOT)/build/cpu/projector/projector_cgls.o \
	$(H2OGPUMLROOT)/build/cpu/elastic_net_ptr.o

CUH2OGPUMLLIB=\
    $(H2OGPUMLROOT)/build/gpu/h2ogpuml_link.o \
	$(H2OGPUMLROOT)/build/gpu/h2ogpumlglm.o \
	$(H2OGPUMLROOT)/build/gpu/h2ogpumlkmeans.o \
	$(H2OGPUMLROOT)/build/gpu/kmeans_labels.o \
	$(H2OGPUMLROOT)/build/gpu/warmstart.o \
	$(H2OGPUMLROOT)/build/gpu/p2pbwcheck.o \
	$(H2OGPUMLROOT)/build/gpu/bwcheck.o \
	$(H2OGPUMLROOT)/build/gpu/matrix/matrix_dense.o \
	$(H2OGPUMLROOT)/build/gpu/matrix/matrix_sparse.o \
	$(H2OGPUMLROOT)/build/gpu/projector/projector_direct_dense.o \
	$(H2OGPUMLROOT)/build/gpu/projector/projector_cgls.o \
	$(H2OGPUMLROOT)/build/gpu/elastic_net_ptr.o

# Python-required CUDA runtime library
PYCUDART=$(CUDAHOME)/libcudart.so	


.PHONY: default all clean cpulib gpulib

default: all

all: cpulib gpulib ch2ogpuml_cpu ch2ogpuml_gpu h2ogpuml_c.o

cpulib:
	$(MAKE) cpulib -C $(H2OGPUMLROOT) IFLAGS=$(IFLAGS)

gpulib:
	$(MAKE) gpulib -C $(H2OGPUMLROOT) IFLAGS=$(IFLAGS)

ch2ogpuml_cpu: h2ogpuml_c.o  Makefile
	touch ch2ogpuml_cpu
	$(MAKE) cpulib -C $(H2OGPUMLROOT) IFLAGS=$(IFLAGS)
	$(CXX) $(CXXFLAGS) -shared -o $@.$(SHARED) $< $(H2OGPUMLLIB) $(LDFLAGS)

ch2ogpuml_gpu: h2ogpuml_c.o Makefile
	touch ch2ogpuml_gpu
	$(MAKE) gpulib -C $(H2OGPUMLROOT) IFLAGS=$(IFLAGS)
	$(CXX) $(CUDFLAGS) -shared -o $@.$(SHARED) $< $(CUH2OGPUMLLIB) $(PYCUDART) $(CULDFLAGS)

h2ogpuml_c.o: h2ogpuml_c.cpp h2ogpuml_c.h Makefile
	$(CXX) $(CXXFLAGS) $< -c -o $@

clean:
	rm -fv *.o *.so *.dylib


