# User Vars
POGSROOT=../../src

# Example Files
EXSRC= lasso.cpp lp_eq.cpp lasso_path.cpp # logistic.cpp lp_ineq.cpp nonneg_l2.cpp svm.cpp

# C++ Flags
CXX=g++
CXXFLAGS=-g -I$(POGSROOT) -fopenmp -std=c++11 -Wall -Wconversion #-O3

# CUDA Flags
CULDFLAGS_=-lcudart -lcublas -lcusparse
CUXXFLAGS=-D__CUDA

# Check System Args.
UNAME = $(shell uname -s)
ifeq ($(UNAME), Darwin)
CULDFLAGS=-L/usr/local/cuda/lib -L/usr/local/lib $(CULDFLAGS_)
else
LDFLAGS=-lm -lopenblas
CULDFLAGS=-L/usr/local/cuda/lib64 $(CULDFLAGS_)
endif

# CPU
cpu: run_all.cpp examples.h $(EXSRC)
	$(MAKE) cpu -C $(POGSROOT)
	$(CXX) $(CXXFLAGS) $(CXXFLAGS) -I$(POGSROOT)/include -o run $(EXSRC) \
  $< $(POGSROOT)/build/pogs.a $(LDFLAGS)

# GPU
gpu: run_all.cpp examples.h $(EXSRC)
	$(MAKE) gpu -C $(POGSROOT)
	$(CXX) $(CXXFLAGS) $(CUXXFLAGS) -I$(POGSROOT)/include -o run $(EXSRC) \
  $< $(POGSROOT)/build/pogs.a $(CULDFLAGS)

clean:
	rm -f *.o *~ *~ run
	rm -rf *.dSYM

