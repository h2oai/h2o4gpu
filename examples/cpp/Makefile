# User Vars
POGSROOT=../../src

# Example Files
EXSRC=lasso_new.cpp #lasso.cpp lasso_path.cpp logistic.cpp lp_eq.cpp lp_ineq.cpp nonneg_l2.cpp svm.cpp

# C++ Flags
CXX=g++
CXXFLAGS=-g -O3 -I$(POGSROOT)/include -fopenmp -std=c++11 -Wall -Wconversion  

# CUDA Flags
CULDFLAGS_=-lcudart -lcublas -lcusparse

# Check System Args.
UNAME = $(shell uname -s)
ifeq ($(UNAME), Darwin)
LDFLAGS=-lm -framework Accelerate
CULDFLAGS=-L/usr/local/cuda/lib -L/usr/local/lib $(CULDFLAGS_)
else
LDFLAGS=-lm
#CULDFLAGS=-L/cm/shared/apps/cuda55/toolkit/current/lib64 $(CULDFLAGS_)
CULDFLAGS=-L/usr/local/cuda/lib64 $(CULDFLAGS_)
endif

# CPU
# cpu: run_all.cpp examples.h $(EXSRC)
# 	$(MAKE) -C $(POGSROOT)
# 	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OOFLAGS) -o run $(EXSRC) $< $(POGSROOT)/pogs.o

# GPU
gpu: run_all.cpp examples.h $(EXSRC)
	$(MAKE) -C $(POGSROOT)
	$(CXX) $(CXXFLAGS) -o run $(EXSRC) $<	$(POGSROOT)/pogs.a $(CULDFLAGS)

clean:
	rm -f *.o *~ *~ run
	rm -rf *.dSYM

