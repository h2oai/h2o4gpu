# User Vars
POGSROOT=../../src

# Example Files
EXSRC=lasso.cpp lasso_path.cpp logistic.cpp lp_eq.cpp lp_ineq.cpp nonneg_l2.cpp svm.cpp

# C++ Flags
CXX=icpc
CXXFLAGS=$(IFLAGS) -g -O3 -I$(POGSROOT)/include -std=c++11 -Wall -Wconversion  -fPIC -openmp -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -liomp5 -lpthread

# CUDA Flags
CULDFLAGS_=-lcudart -lcublas -lcusparse

# Check System Args.
UNAME = $(shell uname -s)
ifeq ($(UNAME), Darwin)
LDFLAGS=-lm -framework Accelerate
CULDFLAGS=-L/usr/local/cuda/lib -L/usr/local/lib $(CULDFLAGS_)
else
LDFLAGS=-lm   -fPIC -openmp -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -liomp5 -lpthread -lm  # -lopenblas
CULDFLAGS=-L/usr/local/cuda/lib64 $(CULDFLAGS_)
endif

# CPU
cpu: run_all.cpp examples.h $(EXSRC)
	$(MAKE) cpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(CXX) $(CXXFLAGS) -o run $(EXSRC) $<	$(POGSROOT)/build/pogs.a $(LDFLAGS)

# GPU
gpu: run_all.cpp examples.h $(EXSRC)
	$(MAKE) gpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(CXX) $(CXXFLAGS) -o run $(EXSRC) $<	$(POGSROOT)/build/pogs.a $(CULDFLAGS)

clean:
	rm -f *.o *~ *~ run
	rm -rf *.dSYM

