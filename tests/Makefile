# How to add new test:
# 1) Add test file (ending with .cu!) in the tests/ directory
# 2) Add the test file name to TESTS variable (without .cu)
# 3) Add a "test.o:" target as shown below, example being "test_kmeans_centroids.o"
#
# How to run:
# All tests: make run_tests
# Single test from test case TestCase named Name: make run_single_test TEST=TestCase.Name
# All tests from test case TestCase: make run_single_test TEST=TestCase*

############### Env variable setup
HOST_OS := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")

ARCH := $(shell getconf LONG_BIT)

ifeq ($(HOST_OS),darwin)
	CXX := clang++
	CUDA_PATH ?= /usr/local/cuda
	LIB_FLAGS := -L$(CUDA_PATH)/lib
else
	CXX := g++
	CUDA_PATH ?= /usr/local/cuda
	LIB_FLAGS_32 := -L$(CUDA_PATH)/lib
	LIB_FLAGS_64 := -L$(CUDA_PATH)/lib64
	LIB_FLAGS := $(LIB_FLAGS_$(ARCH))
endif

GTEST_DIR = googletest/googletest
H2O4GPU_DIR = ..

TESTS = test_kmeans_centroids

GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
                $(GTEST_DIR)/include/gtest/internal/*.h

CUB_HOME=../cub/
CPPFLAGS += -isystem $(GTEST_DIR)/include
CXXFLAGS += -g -Wall -Wextra -pthread --std=c++11

NVCCFLAGS += -g -arch=sm_30 --std=c++11 --expt-extended-lambda
NVCCFLAGS += -lcublas -lcudart -lcudadevrt -lcuda
CUXX=$(CUDA_HOME)/bin/nvcc -L$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/include -I$(CUB_HOME) -lineinfo

## Headers from H2O4GPU required for test compilation
GPU_KMEANS_HDR=../src/gpu/kmeans/kmeans_centroids.h

############### Default target

all : clean test_h2o4gpu

clean :
	rm -f test_h2o4gpu gtest.a gtest_main.a *.o
	rm -rf *.dSYM

############### Builds Google Test lib

GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)

gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest-all.cc

gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest_main.cc

gtest.a : gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

gtest_main.a : gtest-all.o gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

############### Build H2O4GPU tests

# Example for when .cuh/.h/.hpp and .cu/.cpp/.cc files are separated
# test.o: ../src/gpu/kmeans/test.cu ../src/gpu/kmeans/test.cuh $(GPU_KMEANS_HDR)
#	$(CUXX) $(CPPFLAGS) $(NVCCFLAGS) -I$(H2O4GPU_DIR) $< -c -o $@

test_kmeans_centroids.o: gpu/kmeans/test_kmeans_centroids.cu $(GPU_KMEANS_HDR)
	$(CUXX) $(CPPFLAGS) $(NVCCFLAGS) -I$(H2O4GPU_DIR) $< -c -o $@

# Example for when .cuh/.h/.hpp and .cu/.cpp/.cc files are separated
# test_h2o4gpu: $(TESTS:=.o) test.o gtest_main.a

test_h2o4gpu: $(TESTS:=.o) gtest_main.a
	$(CXX) $^ -lpthread -lcublas -lcudart -lcudadevrt -lcuda $(LIB_FLAGS) -o $@

############### Run tests

run_tests: test_h2o4gpu
	./test_h2o4gpu

run_single_test: test_h2o4gpu
	./test_h2o4gpu --gtest_filter=$(TEST)