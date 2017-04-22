# For GPU case, must modify /usr/local/cuda/include/host_config.h to add && __ICC != 1700 to #error about unsupported ICC configuration
ICCFILE := $(shell command -v icpc 2> /dev/null)
MKLROOT := $(shell echo $MKL_ROOT 2> /dev/null)

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


$(warning USEICC is $(USEICC))
$(warning USEMKL is $(USEMKL))

POGSROOT=../

GITHASH := $(shell git rev-parse --short HEAD 2> /dev/null)
HASH=-D_GITHASH_=\"$(GITHASH)\"

MKLSTUFF=-lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -liomp5 -lpthread
ICCSTUFF=-qopenmp
GCCSTUFF=-fopenmp -ggdb3


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

LINKFLAGS=$(ICCSTUFF) $(MKLSTUFF) $(GCCSTUFF)

DFLAGS=-DUSEICC=$(USEICC) -DUSEMKL=$(USEMKL) # -DUSE_NCCL  #-DUSE_NVTX=1 -DDEBUG
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
LDFLAGS=-lm -lblas -lgomp
CULDFLAGS=-L$(CUDAHOME) -lcudart -lcublas -lcusparse -lcuda -lcurand -lgomp $(DFLAGS) #  -lnccl -DUSE_NCCL=1 #  -lnvToolsExt -DUSE_NVTX=1 
SHARED=so
endif

# C++ Flags
CXX=g++
CXXFLAGS=$(IFLAGS) -g -ggdb3 -fopenmp -O0 $(HASH) -I$(POGSROOT)/include -I$(CUDAHOME)/../include -std=c++11 -Wall -Wconversion -fPIC  $(DFLAGS)

# POGS static libraries
POGSLIB=\
	$(POGSROOT)/build/cpu/pogs.o \
	$(POGSROOT)/build/cpu/matrix/matrix_dense.o \
	$(POGSROOT)/build/cpu/matrix/matrix_sparse.o \
	$(POGSROOT)/build/cpu/projector/projector_direct_dense.o \
	$(POGSROOT)/build/cpu/projector/projector_cgls.o \
	$(POGSROOT)/build/cpu/elastic_net_mapd.o

CUPOGSLIB=\
        $(POGSROOT)/build/gpu/pogs_link.o \
	$(POGSROOT)/build/gpu/pogs.o \
	$(POGSROOT)/build/gpu/warmstart.o \
	$(POGSROOT)/build/gpu/p2pbwcheck.o \
	$(POGSROOT)/build/gpu/bwcheck.o \
	$(POGSROOT)/build/gpu/matrix/matrix_dense.o \
	$(POGSROOT)/build/gpu/matrix/matrix_sparse.o \
	$(POGSROOT)/build/gpu/projector/projector_direct_dense.o \
	$(POGSROOT)/build/gpu/projector/projector_cgls.o \
	$(POGSROOT)/build/gpu/elastic_net_mapd.o

# Python-required CUDA runtime library
PYCUDART=$(CUDAHOME)/libcudart.so	


.PHONY: default all
default: all
all: cpogs_cpu cpogs_gpu

cpogs_cpu: pogs_c.o
	$(MAKE) cpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(CXX) $(CXXFLAGS) -shared -o $@.$(SHARED) $< $(POGSLIB) $(LDFLAGS)

cpogs_gpu: pogs_c.o
	$(MAKE) gpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(CXX) $(CUDFLAGS) -shared -o $@.$(SHARED) $< $(CUPOGSLIB) $(PYCUDART) $(CULDFLAGS)

pogs_c.o: pogs_c.cpp pogs_c.h
	$(CXX) $(CXXFLAGS) $< -c -o $@

.PHONY: clean
clean:
	rm -fv *.o *.so *.dylib


