# User Vars
POGSROOT=../../src

# Example Files
EXSRC=lasso.cpp lasso_path.cpp logistic.cpp lp_eq.cpp lp_ineq.cpp nonneg_l2.cpp svm.cpp

# C++ Flags
CXX=g++
CXXFLAGS=-g -O3 -I$(POGSROOT)/include -std=c++11 -Wall -Wconversion -fopenmp

# CUDA Flags
CULDFLAGS_=-lcudart -lcublas -lcusparse

# Check System Args.
UNAME = $(shell uname -s)
ifeq ($(UNAME), Darwin)
LDFLAGS=-lm -framework Accelerate
CULDFLAGS=-L/usr/local/cuda/lib -L/usr/local/lib $(CULDFLAGS_)
else
LDFLAGS=-lm -lopenblas
#CULDFLAGS=-L/cm/shared/apps/cuda55/toolkit/current/lib64 $(CULDFLAGS_)
CULDFLAGS=-L/usr/local/cuda/lib64 $(CULDFLAGS_)
endif

# CPU
cpu: run_all.cpp examples.h $(EXSRC)
	$(MAKE) cpu -C $(POGSROOT)
	$(CXX) $(CXXFLAGS) -o run $(EXSRC) $<	$(POGSROOT)/build/pogs.a $(LDFLAGS)

# GPU
gpu: run_all.cpp examples.h $(EXSRC)
	$(MAKE) gpu -C $(POGSROOT)
	$(CXX) $(CXXFLAGS) -o run $(EXSRC) $<	$(POGSROOT)/build/pogs.a $(CULDFLAGS)

clean:
	rm -f *.o *~ *~ run
	rm -rf *.dSYM

