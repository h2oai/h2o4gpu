# User Vars
GSLROOT=/usr/local

# C++ Flags
CXX=g++
CXXFLAGS=-g -O3 -Wall -Wconversion -std=c++11 -I$(GSLROOT)/include #-fopenmp

# CUDA Flags
CUXX=nvcc
CUFLAGS=-arch=sm_20 -lineinfo
CULDFLAGS=-lcudart -lcublas

# Check System Args.
UNAME = $(shell uname -s)
ifeq ($(UNAME), Darwin)
LDFLAGS=-lgsl -framework Accelerate -lm
CULDFLAGS_=-L/usr/local/cuda/lib -L/usr/local/lib $(CULDFLAGS)
else
LDFLAGS=-lgsl -lgslcblas -lm
CULDFLAGS_=-L/cm/shared/apps/cuda55/toolkit/current/lib64 $(CULDFLAGS)
endif

# CPU
pogs.o: pogs.cpp pogs.hpp prox_lib.hpp
	$(CXX) $(CXXFLAGS) $(IFLAGS) $< -c -o $@

# GPU
pogs_cu_link.o: pogs_cu.o
	$(CUXX) $(CUFLAGS) $< -dlink -o $@

pogs_cu.o: pogs.cu pogs.hpp prox_lib.hpp
	$(CUXX) $(CUFLAGS) $(IFLAGS) $< -dc -o $@

clean:
	rm -f *.o *~ *~
	rm -rf *.dSYM

