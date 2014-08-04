# User Vars
GSLROOT=/usr/local
POGSROOT=../../src

# Example Files
EXSRC=lasso.cpp lp_eq.cpp lp_ineq.cpp nonneg_l2.cpp svm.cpp

# C++ Flags
CXX=g++
CXXFLAGS=-g -O3 -Wall -Wconversion -std=c++11 -I$(GSLROOT)/include -I$(POGSROOT) -fopenmp

# CUDA Flags
CULDFLAGS=-lcudart -lcublas

# Check System Args.
UNAME = $(shell uname -s)
ifeq ($(UNAME), Darwin)
LDFLAGS=-lm -framework Accelerate
CULDFLAGS_=-L/usr/local/cuda/lib -L/usr/local/lib $(CULDFLAGS)
else
LDFLAGS=-lm
CULDFLAGS_=-L/cm/shared/apps/cuda55/toolkit/current/lib64 $(CULDFLAGS)
endif

# CPU
cpu: run_all.cpp examples.h $(EXSRC)
	$(MAKE) -C $(POGSROOT) pogs.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OOFLAGS) -o run $(EXSRC) $< $(POGSROOT)/pogs.o

# GPU
gpu: run_all.cpp examples.h $(EXSRC)
	$(MAKE) -C $(POGSROOT) pogs_cu.o pogs_cu_link.o
	$(CXX) $(CXXFLAGS) $(CULDFLAGS_) -o run \
	$(EXSRC) $< $(POGSROOT)/pogs_cu.o $(POGSROOT)/pogs_cu_link.o

clean:
	rm -f *.o *~ *~ run
	rm -rf *.dSYM

