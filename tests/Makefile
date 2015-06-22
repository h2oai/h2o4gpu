# User Vars
POGSROOT=../src

# Example Files
EXSRC=test_cone.cpp

# C++ Flags
CXX=g++
CXXFLAGS=$(IFLAGS) -g -O3 -I$(POGSROOT)/include -std=c++11 -Wall -Wconversion

# CUDA Flags
CULDFLAGS=-lcudart -lcublas -lcusparse

# Check System Args.
UNAME = $(shell uname -s)
ifeq ($(UNAME), Darwin)
LDFLAGS=-lm -framework Accelerate
CULDFLAGS+=-L/usr/local/cuda/lib -L/usr/local/lib
CXXFLAGS+=-DPOGS_PRETTY_PRINT
IFLAGS+=-DPOGS_PRETTY_PRINT
else
LDFLAGS=-lm -lopenblas
CULDFLAGS+=-L/usr/local/cuda/lib64
endif

# CPU
cpu: $(EXSRC)
	$(MAKE) cpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(CXX) $(CXXFLAGS) -o run $< \
	$(POGSROOT)/build/pogs.a $(LDFLAGS)

# GPU
gpu: $(EXSRC)
	$(MAKE) gpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(CXX) $(CXXFLAGS) -o run $< \
	$(POGSROOT)/build/pogs.a $(CULDFLAGS)

clean:
	rm -f *.o *~ *~ run
	rm -rf *.dSYM
	$(MAKE) clean -C $(POGSROOT)

