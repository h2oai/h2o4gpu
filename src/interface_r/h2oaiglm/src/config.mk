# Change this target between cpu and gpu
TARGET=gpu

# the R header dir, and the R shared library dir on your system
R_PATH := $(shell R RHOME)
R_INC := $(shell R CMD config --cppflags)
R_LIB := $(shell R CMD config --ldflags)

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
    R_FRAMEWORK := -F$(R_PATH)/.. -framework R
    RPATH := -rpath $(CUDA_LIB)
endif

################################################################################
############################# H2OAIGLM specific begin ##############################
################################################################################

#compiler/preprocessor options
HDR=-Iinclude $(R_INC) -I$(CUDA_HOME)/include

#linker options
CXXFLAGS+=$(DEVICEOPTS)  -DH2OAIGLM_SINGLE=0


ifeq ($(TARGET), gpu)
    LD_FLAGS=$(R_LIB) -L"$(CUDA_LIB)" -lcudart -lcublas -lcusparse -lnvToolsExt # -lnccl
else
    LD_FLAGS=blas2cblas.cpp $(shell R CMD config BLAS_LIBS)
endif
LD_FLAGS+=$(R_FRAMEWORK)

h2oaiglm.so: $(TARGET) h2oaiglm_r.o
	$(CXX) -o $@ -shared h2oaiglm_r.o $(OBJDIR)/$(TARGET)/h2oaiglm.a \
	$(RPATH) $(LD_FLAGS) $(CXXFLAGS)

blas2cblas.o: blas2cblas.cpp
	$(CXX) $(CXXFLAGS) $< -c -o $@

h2oaiglm_r.o: h2oaiglm_r.cpp
	$(CXX) $(HDR) $(CXXFLAGS) $< -c -o $@

################################################################################
############################## H2OAIGLM specific end ###############################
################################################################################
