# Instructions
# 1. To build with openmp set IFLAGS=-fopenmp

# 2. Switch target between cpu/gpu
TARGET=cpu

# User Vars
POGSROOT=../

# C++ Flags
CXX=g++
CXXFLAGS=$(IFLAGS) -g -O3 -I$(POGSROOT)/include -std=c++11 -Wall -Wconversion -fPIC


# 3. Set CUDA_HOME= path/to/CUDA/libraries/
ifndef CUDA_HOME
CUDA_HOME=/usr/local/cuda/
endif

# Check System Args.
ifeq ($(shell uname -s), Darwin)
LDFLAGS=-lm -framework Accelerate
CULDFLAGS=-L$(CUDAHOME)/lib -L/usr/local/lib $(CULDFLAGS_)
SHARED=dylib
else
LDFLAGS=-lm -lopenblas
CULDFLAGS=-L$(CUDAHOME)/lib64 -lcudart -lcublas -lcusparse
SHARED=so
endif


# POGS static libraries
POGSLIB=\
	$(POGSROOT)/build/cpu/pogs.o \
	$(POGSROOT)/build/cpu/matrix/matrix_dense.o \
	$(POGSROOT)/build/cpu/matrix/matrix_sparse.o \
	$(POGSROOT)/build/cpu/projector/projector_direct_dense.o \
	$(POGSROOT)/build/cpu/projector/projector_cgls.o

CUPOGSLIB=\
	$(POGSROOT)/build/pogs_link.o \
	$(POGSROOT)/build/gpu/pogs.o \
	$(POGSROOT)/build/gpu/matrix/matrix_dense.o \
	$(POGSROOT)/build/gpu/matrix/matrix_sparse.o \
	$(POGSROOT)/build/gpu/projector/projector_direct_dense.o \
	$(POGSROOT)/build/gpu/projector/projector_cgls.o


.PHONY: default
default: pogs_c.o

libpogs: pogs_c.o
	$(MAKE) cpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(CXX) -shared -o $@.$(SHARED) $< $(POGSLIB) $(LDFLAGS)

libpogs_gpu: pogs_c.o
	$(MAKE) gpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(CXX) -shared -o $@.$(SHARED) $< $(CUPOGSLIB) $(CULDFLAGS)

pogs_c.o: pogs_c.cpp pogs_c.h
	$(CXX) $(CXXFLAGS) $< -c -o $@

.PHONY: clean
clean:
	rm -fv *.o *.so *.dylib

