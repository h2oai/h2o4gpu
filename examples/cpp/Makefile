# User Vars
POGSROOT=../../src

# Example Files
EXSRC=lasso.cpp elastic_net.cpp
EXSRC2=lasso.cpp elastic_net.cpp

# C++ Flags
CXX=g++
CXXFLAGS=$(IFLAGS) -g -O3 -I$(POGSROOT)/include -std=c++11 -Wall -Wconversion -fopenmp #-DUSE_NCCL=1 #-DDEBUG  #-DDEBUG


# CUDA Flags
CUDA_LIB=$(CUDA_HOME)/lib64
CULDFLAGS_=-lcudart -lcublas -lcusparse -lnvToolsExt -lrt -I$(CUDA_HOME)/include  -lnccl -lcuda -lcurand #-DUSE_NCCL=1 #-DUSE_NVTX=1 -DCUDCHECKERR -DDEBUG #

# Check System Args.
UNAME = $(shell uname -s)
ifeq ($(UNAME), Darwin)
LDFLAGS=-lm -framework Accelerate
CULDFLAGS=-L/usr/local/cuda/lib -L/usr/local/lib $(CULDFLAGS_)
else
LDFLAGS=-lm -lblas
CULDFLAGS=-L/usr/local/cuda/lib64 $(CULDFLAGS_)
endif

all: gpualt

default: gpualt

# CPU
cpu: run_all.cpp examples.h $(EXSRC)
	$(MAKE) cpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(CXX) $(CXXFLAGS) -o run $(EXSRC) $<	$(POGSROOT)/build/pogs.a $(LDFLAGS)

cpualt: run_all.cpp examples.h $(EXSRC2)
	$(MAKE) cpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(CXX) $(CXXFLAGS) -o h2oai-glm-cpu $(EXSRC2)  $<	$(POGSROOT)/build/pogs.a $(LDFLAGS)

# GPU
gpu: run_all.cpp examples.h $(EXSRC)
	$(MAKE) gpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(CXX) $(CXXFLAGS) -o run $(EXSRC) $<	$(POGSROOT)/build/pogs.a  $(CULDFLAGS)

gpualt: run_all.cpp examples.h $(EXSRC2)
	$(MAKE) gpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(CXX) -DHAVECUDA $(CXXFLAGS) -o h2oai-glm-gpu $(EXSRC2)  $<	$(POGSROOT)/build/pogs.a $(CULDFLAGS)


clean:
	rm -f *.o *~ *~ run
	rm -rf *.dSYM

