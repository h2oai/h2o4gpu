# Instructions
# 1. To build with openmp set 
# IFLAGS=-fopenmp

# 2. Set POGSROOT= path/to/POGS/src/
POGSROOT=../


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
LDFLAGS=-lm -lblas -lgomp
CULDFLAGS=-L$(CUDAHOME) -lcudart -lcublas -lcusparse -lcuda -lcurand -lgomp #  -lnccl -DUSE_NCCL=1 #  -lnvToolsExt -DUSE_NVTX=1 -DCUDCHEC\
KERR -DDEBUG # 
SHARED=so
endif

# C++ Flags
CXX=g++
CXXFLAGS=$(IFLAGS) -g -ggdb3 -fopenmp -O3 -I$(POGSROOT)/include -I$(CUDAHOME)/../include -std=c++11 -Wall -Wconversion -fPIC  #-DUSE_NCCL   #-DUSE_NVTX=1 -DDEBUG # -Wconversion 

# POGS static libraries
POGSLIB=\
	$(POGSROOT)/build/cpu/pogs.o \
	$(POGSROOT)/build/cpu/matrix/matrix_dense.o \
	$(POGSROOT)/build/cpu/matrix/matrix_sparse.o \
	$(POGSROOT)/build/cpu/projector/projector_direct_dense.o \
	$(POGSROOT)/build/cpu/projector/projector_cgls.o

 #$(POGSROOT)/build/pogs_link.o
CUPOGSLIB=\
	$(POGSROOT)/build/gpu/pogs.o \
	$(POGSROOT)/build/gpu/warmstart.o \
	$(POGSROOT)/build/gpu/p2pbwcheck.o \
	$(POGSROOT)/build/gpu/bwcheck.o \
	$(POGSROOT)/build/gpu/matrix/matrix_dense.o \
	$(POGSROOT)/build/gpu/matrix/matrix_sparse.o \
	$(POGSROOT)/build/gpu/projector/projector_direct_dense.o \
	$(POGSROOT)/build/gpu/projector/projector_cgls.o

# Python-required CUDA runtime library
PYCUDART=$(CUDAHOME)/libcudart.so	


.PHONY: default all
default: pogs_c.o
all: libpogs libpogs_gpu pypogs_gpu

libpogs: pogs_c.o
	$(MAKE) cpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(CXX) -shared -o $@.$(SHARED) $< $(POGSLIB) $(LDFLAGS)

libpogs_gpu: pogs_c.o
	$(MAKE) gpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(CXX) -shared -o $@.$(SHARED) $< $(CUPOGSLIB) $(CULDFLAGS)

pypogs_cpu: pogs_c.o
	$(MAKE) cpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(CXX) -shared -o $@.$(SHARED) $< $(POGSLIB) $(LDFLAGS)

pypogs_gpu: pogs_c.o
	$(MAKE) gpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(CXX) -shared -o $@.$(SHARED) $< $(CUPOGSLIB) $(PYCUDART) $(CULDFLAGS)

pogs_c.o: pogs_c.cpp pogs_c.h
	$(CXX) $(CXXFLAGS) $< -c -o $@

.PHONY: clean
clean:
	rm -fv *.o *.so *.dylib


