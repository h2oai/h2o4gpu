UTEST_ROOT=../tests/cpp
UTEST_OBJ_ROOT=$(UTEST_ROOT)/build
UNITTEST=$(UTEST_OBJ_ROOT)/cpp_tests

HOST_OS := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")

ARCH := $(shell getconf LONG_BIT)

ifeq ($(HOST_OS),darwin)
	CUDA_PATH ?= /usr/local/cuda
	LIB_FLAGS := $(CUDA_PATH)/lib
else
	CUDA_PATH ?= /usr/local/cuda
	LIB_FLAGS_32 := $(CUDA_PATH)/lib
	LIB_FLAGS_64 := $(CUDA_PATH)/lib64
	LIB_FLAGS := $(LIB_FLAGS_$(ARCH))
endif

############### Builds Google Test lib

GTEST_PATH = $(UTEST_ROOT)/googletest/googletest
GTEST_HEADERS = $(GTEST_PATH)/include/gtest/*.h \
                $(GTEST_PATH)/include/gtest/internal/*.h

GTEST_SRCS_ = $(GTEST_PATH)/src/*.cc $(GTEST_PATH)/src/*.h $(GTEST_HEADERS)

GTEST_CPPFLAGS=-isystem $(GTEST_PATH)/include

$(UTEST_OBJ_ROOT)/gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(GTEST_CPPFLAGS) -I$(GTEST_PATH) $(CXXFLAGS) -c \
	        $(GTEST_PATH)/src/gtest-all.cc -o $@

$(UTEST_OBJ_ROOT)/gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(GTEST_CPPFLAGS) -I$(GTEST_PATH) $(CXXFLAGS) -c \
            $(GTEST_PATH)/src/gtest_main.cc -o $@

$(UTEST_OBJ_ROOT)/gtest_main.a : $(UTEST_OBJ_ROOT)/gtest-all.o $(UTEST_OBJ_ROOT)/gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

############### H2O4GPU Unit Tests

UNITTEST_SRC=$(wildcard $(UTEST_ROOT)/*.cu $(UTEST_ROOT)/*/*.cu $(UTEST_ROOT)/*/*/*.cu)
UNITTEST_OBJ=$(patsubst $(UTEST_ROOT)%.cu, $(UTEST_OBJ_ROOT)%.o, $(UNITTEST_SRC))

GTEST_INC=$(GTEST_PATH)/include/

COVER_OBJ=$(patsubst %.o, %.gcda, $(ALL_OBJ)) $(patsubst %.o, %.gcda, $(UNITTEST_OBJ))

$(UTEST_OBJ_ROOT)/%.o: $(UTEST_ROOT)/%.cu
	@mkdir -p $(@D)
	$(CUXX) $(CUFLAGS) -I$(GTEST_INC) -o $@ -c $<

UTEST_LIBS=-fopenmp -lcudart_static -lcublas_static -lcusolver_static -lcusparse_static -lblas -lcurand_static
UTEST_LD= -L$(CUDA_PATH) -L$(LIB_FLAGS)

ifeq ($(USEcudadevrt),1)
UTEST_LIBS+=-lcudadevrt
endif


UTEST_DEPS=$(OBJDIR)/gpu/h2o4gpu_link.o $(GPU_OBJ) $(GPU_MTX_OBJ) $(GPU_PRJ_OBJ) $(CPU_PRJ_OBJ)

$(UNITTEST): $(UNITTEST_OBJ) $(UTEST_DEPS) $(UTEST_OBJ_ROOT)/gtest_main.a
	$(CXX) $(UTEST_DEPS) $(UNITTEST_OBJ) -o $@ $(UTEST_LIBS) $(UTEST_LD) $(UTEST_OBJ_ROOT)/gtest_main.a -lpthread

ALL_TEST=$(UNITTEST)
ALL_TEST_OBJ=$(UNITTEST_OBJ)

clean_cpp_tests:
	rm -rf $(UTEST_OBJ_ROOT)