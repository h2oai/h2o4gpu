# C++ Flags
CXX=g++
CXXFLAGS=-g -O3 -Wall -Wconversion -std=c++11  -fPIC -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread

pogs_c.o: pogs_c.cpp pogs_c.h
	$(CXX) $(CXXFLAGS) -I../include $< -c -o $@

clean:
	rm *.o

