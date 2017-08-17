# User Vars
H2OGPUMLROOT=../../src

ifndef N
N=1
endif

#.NOTPARALLEL:

.PHONY: all default clean run runcpu ../../src/build/cpu/h2ogpuml.a ../../src/build/gpu/h2ogpuml.a

all: cpuall gpuall

default: gpuall

SOURCESORIG=./pogs/lasso.cpp  ./pogs/lasso_path2.cpp  ./pogs/lasso_path.cpp  ./pogs/logistic.cpp  ./pogs/lp_eq.cpp  ./pogs/lp_ineq.cpp  ./pogs/nonneg_l2.cpp ./pogs/run_all_orig.cpp  ./pogs/svm.cpp
SOURCESALT=elastic_net.cpp run_all.cpp
SOURCESPTR=elastic_net_ptr_driver.cpp run_all.cpp

../../src/build/cpu/h2ogpuml.a:
	$(MAKE)  cpulib -C $(H2OGPUMLROOT) IFLAGS=$(IFLAGS)

../../src/build/gpu/h2ogpuml.a:
	$(MAKE)  gpulib -C $(H2OGPUMLROOT) IFLAGS=$(IFLAGS)

# CPU
cpuall: Makefile Makefile.cpu $(SOURCESORIG) $(SOURCESALT) $(SOURCESPTR) ../../src/build/cpu/h2ogpuml.a
	touch cpuall
	$(MAKE)  cpulib -C $(H2OGPUMLROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.cpu h2ogpuml-glm-cpu-orig h2ogpuml-glm-cpu h2ogpuml-glm-cpu-ptr h2ogpuml-kmeans-cpu

h2ogpuml-glm-cpu-orig: Makefile Makefile.cpu $(SOURCESORIG) ../../src/build/cpu/h2ogpuml.a
	$(MAKE)  cpulib -C $(H2OGPUMLROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.cpu h2ogpuml-glm-cpu-orig

h2ogpuml-glm-cpu: Makefile Makefile.cpu $(SOURCESALT) ../../src/build/cpu/h2ogpuml.a
	$(MAKE)  cpulib -C $(H2OGPUMLROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.cpu h2ogpuml-glm-cpu

h2ogpuml-glm-cpu-ptr: Makefile Makefile.cpu $(SOURCESPTR)  ../../src/build/cpu/h2ogpuml.a
	$(MAKE)  cpulib -C $(H2OGPUMLROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.cpu h2ogpuml-glm-cpu-ptr

h2ogpuml-kmeans-cpu: Makefile Makefile.cpu kmeans_driver.cpp ../../src/build/cpu/h2ogpuml.a
	$(MAKE)  cpulib -C $(H2OGPUMLROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.cpu h2ogpuml-kmeans-cpu

# GPU
gpuall: Makefile Makefile.gpu $(SOURCESORIG) $(SOURCESALT) $(SOURCESPTR) ../../src/build/gpu/h2ogpuml.a
	touch gpuall
	$(MAKE)  gpulib -C $(H2OGPUMLROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.gpu h2ogpuml-glm-gpu-orig h2ogpuml-glm-gpu h2ogpuml-glm-gpu-ptr h2ogpuml-kmeans-gpu

h2ogpuml-glm-gpu-orig: Makefile Makefile.gpu $(SOURCESORIG)  ../../src/build/gpu/h2ogpuml.a
	$(MAKE)  gpulib -C $(H2OGPUMLROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.gpu h2ogpuml-glm-gpu-orig

h2ogpuml-glm-gpu: Makefile Makefile.gpu $(SOURCESALT) ../../src/build/gpu/h2ogpuml.a
	$(MAKE)  gpulib -C $(H2OGPUMLROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.gpu h2ogpuml-glm-gpu

h2ogpuml-glm-gpu-ptr: Makefile Makefile.gpu $(SOURCESPTR) ../../src/build/gpu/h2ogpuml.a
	$(MAKE)  gpulib -C $(H2OGPUMLROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.gpu h2ogpuml-glm-gpu-ptr

h2ogpuml-kmeans-gpu: Makefile Makefile.gpu kmeans_driver.cpp ../../src/build/gpu/h2ogpuml.a
	$(MAKE)  gpulib -C $(H2OGPUMLROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.gpu h2ogpuml-kmeans-gpu

run: h2ogpuml-glm-gpu-ptr
	./h2ogpuml-glm-gpu-ptr train.txt 0 $(N) $(N) 100 5 8 1 0 0.2

runcpu: h2ogpuml-glm-cpu-ptr
	./h2ogpuml-glm-cpu-ptr train.txt 0 $(N) $(N) 100 5 8 1 0 0.2

clean:
	rm -f *.o *~ *~
	rm -rf h2ogpuml-glm-cpu-orig h2ogpuml-glm-cpu h2ogpuml-glm-cpu-ptr
	rm -rf h2ogpuml-glm-gpu-orig h2ogpuml-glm-gpu h2ogpuml-glm-gpu-ptr
	rm -rf *.dSYM
	rm -rf core
