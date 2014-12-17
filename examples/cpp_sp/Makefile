# User Vars
POGSROOT=../../src

# Example Files
EXSRC=lasso.cpp lp_eq.cpp lasso_path.cpp # logistic.cpp lp_ineq.cpp nonneg_l2.cpp svm.cpp

# C++ Flags
CXX=g++
CXXFLAGS=-g -O3 -I$(POGSROOT) -fopenmp -std=c++11 -Wall -Wconversion  

# CUDA Flags
CULDFLAGS_=-lcudart -lcublas -lcusparse
CUXXFLAGS=-D__CUDA

# Check System Args.
UNAME = $(shell uname -s)
ifeq ($(UNAME), Darwin)
CULDFLAGS=-L/usr/local/cuda/lib -L/usr/local/lib $(CULDFLAGS_)
else
LDFLAGS=-lm
CULDFLAGS=-L/usr/local/cuda/lib64 $(CULDFLAGS_)
endif

# CPU
# cpu: run_all.cpp examples.h $(EXSRC)
# 	$(MAKE) -C $(POGSROOT) pogs.o
# 	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OOFLAGS) -o run $(EXSRC) $< $(POGSROOT)/pogs.o

# GPU
gpu: run_all.cpp examples.h $(EXSRC)
	$(MAKE) -C $(POGSROOT) pogs_sp_cu.o pogs_cu.o pogs_cu_link.o cml.o
	$(CXX) $(CXXFLAGS) $(CUXXFLAGS) -o run $(EXSRC) $< $(POGSROOT)/pogs_sp_cu.o \
	$(POGSROOT)/pogs_cu_link.o $(POGSROOT)/pogs_cu.o $(POGSROOT)/cml.o $(CULDFLAGS)

clean:
	rm -f *.o *~ *~ run
	rm -rf *.dSYM

