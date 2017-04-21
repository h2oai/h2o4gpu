# For GPU case, must modify /usr/local/cuda/include/host_config.h to add && __ICC != 1700 to #error about unsupported ICC configuration
USEICC=0
USEMKL=1

# Bulid directory
OBJDIR=build

RLIBHOME := $(shell R RHOME)
RINCLUDEHOME := $(shell R CMD config --cppflags)

GITHASH := $(shell git rev-parse --short HEAD 2> /dev/null)
HASH=-D_GITHASH_=\"$(GITHASH)\"

LINKFLAGS=-fopenmp

ifeq ($(USEICC),1)
LINKFLAGS=-qopenmp -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -liomp5 -lpthread$
endif

ifeq ($(USEMKL),1)$
LINKFLAGS += -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -liomp5 -lpthread$
ifeq ($(USEICC),1)$
LINKFLAGS += -qopenmp
endif
endif

# C++ Flags
ifeq ($(USEICC),0)
CXX=g++
<<<<<<< Updated upstream
CXXFLAGS=$(IFLAGS) -g -ggdb3 -O3 $(HASH) -Wall -std=c++11 -fPIC $(LINKFLAGS) $(RINCLUDEHOME) #-DUSE_NCCL   #-DUSE_NVTX=1 -DDEBUG # -Wconversion
=======
CXXFLAGS=$(IFLAGS) -g -ggdb3 -O0 $(HASH) -Wall -std=c++11 -fPIC -fopenmp $(RINCLUDEHOME) #-DUSE_NCCL   #-DUSE_NVTX=1 -DDEBUG # -Wconversion
>>>>>>> Stashed changes
else
CXX=icpc
CXXFLAGS=$(IFLAGS) -g -O3 $(HASH) -std=c++11 -Wall -Wconversion -fPIC $(LINKFLAGS) $(RINCLUDEHOME)  #-DUSE_NCCL=1 #-DDEBUG  #-DDEBUG
endif

# CUDA Flags
NVCC_GENCODE ?= -gencode=arch=compute_35,code=sm_35 \
                -gencode=arch=compute_50,code=sm_50 \
                -gencode=arch=compute_52,code=sm_52 \
                -gencode=arch=compute_52,code=compute_52

CUDA_LIB=$(CUDA_HOME)/lib64
CUDA_VERSION ?= $(shell ls $(CUDA_LIB)/libcudart.so.* | head -1 | rev | cut -d "." -f -2 | rev)
CUDA_MAJOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 1)
CUDA_MINOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 2)

ifeq ($(USEICC),0)
OPENMP=-fopenmp
NVCCEXTRA=
else
OPENMP=-qopenmp
NVCCEXTRA=-ccbin=icpc
endif

# Assume installed nccl into $(CUDA_HOME) and moved libraries in lib to lib64
CUXX=$(CUDA_HOME)/bin/nvcc $(NVCCEXTRA) -L$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/include
CUFLAGS=$(IFLAGS) -std=c++11 -g -O0 $(HASH) -Xcompiler -fPIC -Xcompiler $(OPENMP) $(RINCLUDEHOME) # -DUSE_NCCL  #-DUSE_NVTX=1 -DDEBUG -DCUDCHECKERR #
CUFLAGS += -DCUDA_MAJOR=$(CUDA_MAJOR) -DCUDA_MINOR=$(CUDA_MINOR)
#-gencode arch=compute_20,code=sm_20

# Load local (e.g. R) specific config
include config.mk

# POGS header files.
POGS_HDR=\
	include/interface_defs.h \
	include/pogs.h \
	include/prox_lib.h \
	include/util.h \
	include/matrix/matrix.h \
	include/matrix/matrix_dense.h \
	include/matrix/matrix_sparse.h \
	include/projector/projector_cgls.h \
	include/projector/projector_direct.h

# CPU Specific headers and object files.
GSL_HDR=\
	cpu/include/gsl/cblas.h \
	cpu/include/gsl/gsl_blas.h \
	cpu/include/gsl/gsl_linalg.h \
	cpu/include/gsl/gsl_matrix.h \
	cpu/include/gsl/gsl_rand.h \
	cpu/include/gsl/gsl_spblas.h \
	cpu/include/gsl/gsl_spmat.h \
	cpu/include/gsl/gsl_vector.h

CPU_HDR=\
	cpu/include/cgls.h \
	cpu/include/equil_helper.h \
	cpu/include/projector_helper.h
CPU_MTX_OBJ=\
	$(OBJDIR)/cpu/matrix/matrix_sparse.o \
	$(OBJDIR)/cpu/matrix/matrix_dense.o
CPU_PRJ_OBJ=\
	$(OBJDIR)/cpu/projector/projector_cgls.o \
	$(OBJDIR)/cpu/projector/projector_direct_dense.o
CPU_OBJ=$(OBJDIR)/cpu/pogs.o $(OBJDIR)/common/elastic_net_mapd.o 

# GPU Specific headers and object files.
CML_HDR=\
	gpu/include/cml/cblas.h \
	gpu/include/cml/cml_blas.cuh \
	gpu/include/cml/cml_defs.cuh \
	gpu/include/cml/cml_linalg.cuh \
	gpu/include/cml/cml_matrix.cuh \
	gpu/include/cml/cml_rand.cuh \
	gpu/include/cml/cml_spblas.cuh \
	gpu/include/cml/cml_spmat.cuh \
	gpu/include/cml/cml_utils.cuh \
	gpu/include/cml/cml_vector.cuh

GPU_HDR=\
	gpu/include/cgls.cuh \
	gpu/include/equil_helper.cuh \
	gpu/include/projector_helper.cuh
GPU_MTX_OBJ=\
	$(OBJDIR)/gpu/matrix/matrix_dense.o \
	$(OBJDIR)/gpu/matrix/matrix_sparse.o
GPU_PRJ_OBJ=\
	$(OBJDIR)/gpu/projector/projector_cgls.o \
	$(OBJDIR)/gpu/projector/projector_direct_dense.o
GPU_OBJ=$(OBJDIR)/gpu/pogs.o $(OBJDIR)/gpu/warmstart.o  $(OBJDIR)/gpu/p2pbwcheck.o  $(OBJDIR)/gpu/bwcheck.o $(OBJDIR)/common/elastic_net_mapd.o

# Set vpath for build
VPATH=cpu cpu/matrix cpu/projector gpu gpu/matrix gpu/projector

all: gpu cpu

default: gpu

# Build all
cpu: $(CPU_OBJ) $(CPU_MTX_OBJ) $(CPU_PRJ_OBJ)
	ar cr $(OBJDIR)/cpu/pogs.a $^

gpu: $(OBJDIR)/gpu/pogs_link.o $(GPU_OBJ) $(GPU_MTX_OBJ) $(GPU_PRJ_OBJ)
	ar cr $(OBJDIR)/gpu/pogs.a $^


# Directories
$(OBJDIR):
	mkdir -p $@

# Directories Common
$(OBJDIR)/common: $(OBJDIR)
	mkdir -p $@ 

# Directories CPU
$(OBJDIR)/cpu: $(OBJDIR)
	mkdir -p $@ 

$(OBJDIR)/cpu/matrix: $(OBJDIR)/cpu
	mkdir -p $@ 

$(OBJDIR)/cpu/projector: $(OBJDIR)/cpu
	mkdir -p $@ 

# Directories GPU
$(OBJDIR)/gpu: $(OBJDIR)
	mkdir -p $@ 

$(OBJDIR)/gpu/matrix: $(OBJDIR)/gpu
	mkdir -p $@ 

$(OBJDIR)/gpu/projector: $(OBJDIR)/gpu
	mkdir -p $@ 


# Common objects
$(OBJDIR)/common/elastic_net_mapd.o: common/elastic_net_mapd.cpp $(POGS_HDR) $(GSL_HDR) | $(OBJDIR)/common
	$(CXX) -Iinclude -Icommon/include $< $(CXXFLAGS) -c -o $@ 

# POGS CPU objects
$(OBJDIR)/cpu/pogs.o: cpu/pogs.cpp $(POGS_HDR) $(GSL_HDR) | $(OBJDIR)/cpu
	$(CXX) -Iinclude -Icpu/include $< $(CXXFLAGS) -c -o $@ 

$(OBJDIR)/cpu/matrix/%.o: %.cpp $(CPU_HDR) | $(OBJDIR)/cpu/matrix
	$(CXX) -Iinclude -Icpu/include $< $(CXXFLAGS) $(IFLAGS) -c -o $@

$(OBJDIR)/cpu/projector/%.o: %.cpp $(CPU_HDR) | $(OBJDIR)/cpu/projector
	$(CXX) -Iinclude -Icpu/include $< $(CXXFLAGS) $(IFLAGS) -c -o $@

$(OBJDIR)/cpu/%.o: %.cpp $(CPU_HDR) | $(OBJDIR)/cpu
	$(CXX) -Iinclude -Icpu/include $< $(CXXFLAGS) $(IFLAGS) -c -o $@

# POGS GPU objects
$(OBJDIR)/gpu/pogs_link.o: $(GPU_OBJ) $(GPU_MTX_OBJ) $(GPU_PRJ_OBJ) | $(OBJDIR)
	$(CUXX) $(CUFLAGS) $^ -dlink -o $@ 

$(OBJDIR)/gpu/pogs.o: gpu/pogs.cu $(POGS_HDR) $(CML_HDR) | $(OBJDIR)/gpu
	$(CUXX) -Iinclude -Igpu/include $< $(CUFLAGS) -dc -o $@ 

$(OBJDIR)/gpu/warmstart.o: gpu/warmstart.cu $(POGS_HDR) $(CML_HDR) | $(OBJDIR)/gpu
	$(CUXX) -Iinclude -Igpu/include $< $(CUFLAGS) -dc -o $@ 
$(OBJDIR)/gpu/p2pbwcheck.o: gpu/p2pbwcheck.cu $(POGS_HDR) $(CML_HDR) | $(OBJDIR)/gpu
	$(CUXX) -Iinclude -Igpu/include $< $(CUFLAGS) -dc -o $@ 
$(OBJDIR)/gpu/bwcheck.o: gpu/bwcheck.cu $(POGS_HDR) $(CML_HDR) | $(OBJDIR)/gpu
	$(CUXX) -Iinclude -Igpu/include $< $(CUFLAGS) -dc -o $@ 

$(OBJDIR)/gpu/matrix/%.o: %.cu $(GPU_HDR) | $(OBJDIR)/gpu/matrix
	$(CUXX) -Iinclude -Igpu/include $< $(CUFLAGS) $(IFLAGS) -dc -o $@

$(OBJDIR)/gpu/projector/%.o: %.cu $(GPU_HDR) | $(OBJDIR)/gpu/projector
	$(CUXX) -Iinclude -Igpu/include $< $(CUFLAGS) $(IFLAGS) -dc -o $@

$(OBJDIR)/gpu/%.o: %.cu $(GPU_HDR) | $(OBJDIR)/gpu
	$(CUXX) -Iinclude -Igpu/include $< $(CUFLAGS) $(IFLAGS) -dc -o $@

clean:
	rm -rf pogs.a build/ *.dSYM
	find -name "*.so" -delete
	find -name "*.a" -delete
	find -name "*.o" -delete
