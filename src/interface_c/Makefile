# C++ Flags
CXX=g++
CXXFLAGS=-g -O3 -Wall -Wconversion -std=c++11 

pogs_c.o: pogs_c.cpp pogs_c.h
	$(CXX) $(CXXFLAGS) -I.. $< -c -o $@

clean:
	rm *.o

