# User Vars
H2OAIGLMROOT=../../src

# Example Files
EXSRC= lasso.cpp lp_eq.cpp lasso_path.cpp # logistic.cpp lp_ineq.cpp nonneg_l2.cpp svm.cpp

# C++ Flags
CXX=g++
CXXFLAGS=-g -I$(H2OAIGLMROOT) -fopenmp -std=c++11 -Wall -Wconversion #-O3

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
	$(MAKE) cpu -C $(H2OAIGLMROOT)
	$(CXX) $(CXXFLAGS) $(CXXFLAGS) -I$(H2OAIGLMROOT)/include -o run $(EXSRC) \
  $< $(H2OAIGLMROOT)/build/h2oaiglm.a $(LDFLAGS)

# GPU
gpu: run_all.cpp examples.h $(EXSRC)
	$(MAKE) gpu -C $(H2OAIGLMROOT)
	$(CXX) $(CXXFLAGS) $(CUXXFLAGS) -I$(H2OAIGLMROOT)/include -o run $(EXSRC) \
  $< $(H2OAIGLMROOT)/build/h2oaiglm.a $(CULDFLAGS)

clean:
	rm -f *.o *~ *~ run
	rm -rf *.dSYM

