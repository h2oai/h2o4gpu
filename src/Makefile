# User Vars

# C++ Flags
CXX=g++
CXXFLAGS=-g -O3 -Wall -Wconversion -fPIC

# CUDA Flags
CUXX=nvcc
CUFLAGS=-arch=sm_50 -Xcompiler -fPIC

.PHONY: all cpu gpu
all: pogs.o pogs_cu_link.o pogs_cu.o pogs_sp_cu.o cml.o
cpu: pogs.o
gpu: pogs_cu_link.o pogs_cu.o pogs_sp_cu.o cml.o 

# CPU
pogs.o: pogs.cpp pogs.h prox_lib.h _interface_defs.h sinkhorn_knopp.h
	$(CXX) $(CXXFLAGS) $(IFLAGS) $< -c -o $@

# GPU
pogs_cu_link.o: pogs_cu.o pogs_sp_cu.o cml.o
	$(CUXX) $(CUFLAGS) $^ -dlink -o $@

pogs_cu.o: pogs.cu pogs.h prox_lib.h
	$(CUXX) $(CUFLAGS) $(IFLAGS) $< -dc -o $@

pogs_sp_cu.o: pogs_sp.cu pogs.h prox_lib.h
	$(CUXX) $(CUFLAGS) $(IFLAGS) $< -dc -o $@

cml.o: cml/cml_blas.cu cml/cml_blas.cuh
	$(CUXX) -Icml $(CUFLAGS) $< -dc -o $@

.PHONY: clean
clean:
	rm -f *.o *~
	rm -rf *.dSYM

