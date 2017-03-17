# Change this target between cpu and gpu
TARGET=gpu

# set R_HOME, R_INC, and R_LIB to the the R install dir,
# the R header dir, and the R shared library dir on your system
R_HOME := $(shell R RHOME)
R_INC := $(R_HOME)/include
R_LIB := $(R_HOME)/lib

# replace these three lines with
# CUDA_HOME := <path to your cuda install>
ifndef CUDA_HOME
    CUDA_HOME := /usr/local/cuda
endif

ARCH := $(shell uname -m)

# replace these five lines with
# CUDA_LIB := <path to your cuda shared libraries>
ifeq ($(ARCH), i386)
    CUDA_LIB := $(CUDA_HOME)/lib64
else
    CUDA_LIB := $(CUDA_HOME)/lib64
endif

OS := $(shell uname -s)
ifeq ($(OS), Darwin)
    ifeq ($(ARCH), x86_64)
        DEVICEOPTS := -m64
    endif
    CUDA_LIB := $(CUDA_HOME)/lib64
    R_FRAMEWORK := -F$(R_HOME)/.. -framework R
    RPATH := -rpath $(CUDA_LIB)
endif

################################################################################
############################# POGS specific begin ##############################
################################################################################

#compiler/preprocessor options
HDR=-Iinclude -I"$(R_INC)"

#linker options
CXXFLAGS+=$(DEVICEOPTS)  -DPOGS_SINGLE=0
ifeq ($(TARGET), gpu)
    LD_FLAGS=-L"$(R_LIB)" -L"$(CUDA_LIB)" -lcudart -lcublas -lcusparse
else
    LD_FLAGS=blas2cblas.cpp $(shell R CMD config BLAS_LIBS)
endif
LD_FLAGS+=$(R_FRAMEWORK)

pogs.so: $(TARGET) pogs_r.o
	$(CXX) -o $@ -shared pogs_r.o $(OBJDIR)/pogs.a \
	$(RPATH) $(LD_FLAGS) $(CXXFLAGS)

blas2cblas.o: blas2cblas.cpp
	$(CXX) $(CXXFLAGS) $< -c -o $@

pogs_r.o: pogs_r.cpp
	$(CXX) $(HDR) $(CXXFLAGS) $< -c -o $@

################################################################################
############################## POGS specific end ###############################
################################################################################
