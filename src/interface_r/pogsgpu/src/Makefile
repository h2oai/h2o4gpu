include config.mk

#compiler/preprocessor options
INCS := -I. -I"$(CUDA_INC)" -I"$(R_INC)"
PARAMS := $(DEVICEOPTS) -Xcompiler $(CPICFLAGS)
CXXFLAGS := -stdlib=libstdc++

#linker options
LD_PARAMS_CU := $(DEVICEOPTS) -Xlinker '$(RPATH) $(R_FRAMEWORK)'
LD_PARAMS := $(DEVICEOPTS) $(R_FRAMEWORK)
LIBS :=  -L"$(R_LIB)" -L"$(CUDA_LIB)" -lcudart -lcublas $(shell R CMD config BLAS_LIBS)

TARGETS := pogsgpu.so

NVCC := $(CUDA_HOME)/bin/nvcc -arch=sm_20
CXX := g++

all: $(TARGETS)

$(TARGETS): pogs_link.o pogs.o pogs_r.o
	$(CXX) -shared $(LIBS) $(RPATH) $(LD_PARAMS) $(CXXFLAGS) $^ -o $@

pogs_r.o: pogs_r.cpp pogs.h matrix_util.h
	$(CXX) $(INCS) $(LD_PARAMS) $(CPICFLAGS) $(CXXFLAGS) $< -c -o $@

pogs_link.o: pogs.o
	$(NVCC) $(LD_PARAMS_CU) $(LIBS) $< -dlink -o $@

pogs.o: pogs.cu pogs.h prox_lib.h
	$(NVCC) $(INCS) $(PARAMS) $(IFLAGS) -D__R__ $< -dc -o $@

clean:
	rm -rf *.o

.PHONY: all clean
