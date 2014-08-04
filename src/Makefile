# User Vars
#GSLROOT=/usr/local

# C++ Flags
CXX=g++
CXXFLAGS=-g -O3 -Wall -Wconversion

# CUDA Flags
CUXX=nvcc
CUFLAGS=-arch=sm_20

# CPU
pogs.o: pogs.cpp pogs.h prox_lib.h _interface_defs.h sinkhorn_knopp.h
	$(CXX) -I. $(CXXFLAGS) $(IFLAGS) $< -c -o $@

# GPU
pogs_cu_link.o: pogs_cu.o
	$(CUXX) $(CUFLAGS) $< -dlink -o $@

pogs_cu.o: pogs.cu pogs.h prox_lib.h
	$(CUXX) -I. $(CUFLAGS) $(IFLAGS) $< -dc -o $@

test:
	$(MAKE) all -C ./unit 

clean:
	rm -f *.o *~ *~
	rm -rf *.dSYM

