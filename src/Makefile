# # C++ Flags
# CXX=g++
# CXXFLAGS=-g -O3 -Wall -Wconversion
# 
# # CUDA Flags
# CUXX=nvcc
# CUFLAGS=-arch=sm_20 -Xcompiler -fPIC
# 
# all: pogs.o pogs_cu_link.o pogs_cu.o pogs_sp_cu.o cml.o
# 
# # CPU
# pogs.o: pogs.cpp pogs.h prox_lib.h _interface_defs.h sinkhorn_knopp.h
# 	$(CXX) $(CXXFLAGS) $(IFLAGS) $< -c -o $@
# 
# # GPU
# pogs_cu_link.o: pogs_cu.o pogs_sp_cu.o cml.o
# 	$(CUXX) $(CUFLAGS) $^ -dlink -o $@
# 
# pogs_cu.o: pogs.cu pogs.h prox_lib.h
# 	$(CUXX) $(CUFLAGS) $(IFLAGS) $< -dc -o $@
# 
# pogs_sp_cu.o: pogs_sp.cu pogs.h prox_lib.h
# 	$(CUXX) $(CUFLAGS) $(IFLAGS) $< -dc -o $@
# 
# cml.o: cml/cml_blas.cu cml/cml_blas.cuh
# 	$(CUXX) -Icml $(CUFLAGS) $< -dc -o $@
# 
# clean:
# 	rm -f *.o *~
# 	rm -rf *.dSYM

# C++ Flags
CXX=g++
CXXFLAGS=-g -O3 -Wall -Wconversion

# CUDA Flags
CUXX=nvcc
CUFLAGS=-arch=sm_20 -Xcompiler -fPIC -DDEBUG

MATRIX=\
	include/matrix/matrix.h \
	include/matrix/matrix_dense.h \
	include/matrix/matrix_sparse.h
PROJECTOR=\
	include/projector/projector_cgls.h \
	include/projector/projector_direct.h
POGS_HDR=\
	include/interface_defs.h \
	include/pogs.h \
	include/prox_lib.h \
	$(MATRIX) $(PROJECTOR)

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
CML_OBJ=gpu/cml/cml_blas.o

GPU_HDR=\
	gpu/include/cgls.cuh \
	gpu/include/equil_helper.cuh \
	gpu/include/projection_helper.cuh \
	gpu/include/util.cuh
GPU_MTX_OBJ=\
	gpu/matrix/matrix_dense.o \
  gpu/matrix/matrix_sparse.o
GPU_PRJ_OBJ=\
	gpu/projector/projector_cgls.o \
	gpu/projector/projector_direct_dense.o
GPU_OBJ=gpu/pogs.o

VPATH=gpu:gpu/matrix:gpu/cml:gpu/projector

all: pogs_link.o $(GPU_OBJ) $(GPU_MTX_OBJ) $(GPU_PRJ_OBJ) $(CML_OBJ)
	ar cr pogs.a $^
 
# # CPU
# pogs.o: pogs.cpp pogs.h prox_lib.h _interface_defs.h sinkhorn_knopp.h
# 	$(CXX) $(CXXFLAGS) $(IFLAGS) $< -c -o $@
# 
# GPU
#

# TODO: Build directory!!!!
# POGS GPU
pogs_link.o: $(GPU_OBJ) $(GPU_MTX_OBJ) $(GPU_PRJ_OBJ) $(CML_OBJ)
	$(CUXX) $(CUFLAGS) $^ -dlink -o $@

gpu/pogs.o: gpu/pogs.cu $(POGS_HDR) $(CML_HDR)
	$(CUXX) -Iinclude -Igpu/include $< $(CUFLAGS) -dc -o $@

gpu/matrix/%.o: %.cu $(GPU_HDR)
	$(CUXX) -Iinclude -Igpu/include $< $(CUFLAGS) $(IFLAGS) -dc -o $@

gpu/projector/%.o: %.cu $(GPU_HDR)
	$(CUXX) -Iinclude -Igpu/include $< $(CUFLAGS) $(IFLAGS) -dc -o $@

gpu/cml/%.o: %.cu $(GPU_HDR) $(CML_HDR)
	$(CUXX) -Iinclude -Igpu/include $< $(CUFLAGS) $(IFLAGS) -dc -o $@

gpu/%.o: %.cu $(GPU_HDR)
	$(CUXX) -Iinclude -Igpu/include $< $(CUFLAGS) $(IFLAGS) -dc -o $@

clean:
	rm -f *.a *.o gpu/*.o gpu/matrix/*.o gpu/projector/*.o gpu/cml/*.o
	rm -rf *.dSYM

