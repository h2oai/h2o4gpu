location = $(CURDIR)/$(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))
WHERE := $(location)
$(info ** -> $(WHERE))
$(info ** ------------------------------------------------------------------ **)

NVCC := $(shell command -v nvcc 2> /dev/null)

# User Vars
H2O4GPUROOT=../../src

ifndef N
N=1
endif

#.NOTPARALLEL:

ifdef NVCC
.PHONY: all default clean run runcpu ../../src/build/cpu/h2o4gpu.a ../../src/build/gpu/h2o4gpu.a
all: cpuall gpuall
default: gpuall
else
.PHONY: all default clean run runcpu ../../src/build/cpu/h2o4gpu.a
all: cpuall
default: cpuall
endif

SOURCESALT=elastic_net.cpp run_all.cpp
SOURCESPTR=elastic_net_ptr_driver.cpp run_all.cpp

../../src/build/cpu/h2o4gpu.a:
	$(MAKE)  cpulib -C $(H2O4GPUROOT) IFLAGS=$(IFLAGS)

../../src/build/gpu/h2o4gpu.a:
	$(MAKE)  gpulib -C $(H2O4GPUROOT) IFLAGS=$(IFLAGS)

# CPU
cpuall: Makefile Makefile.cpu $(SOURCESALT) $(SOURCESPTR) ../../src/build/cpu/h2o4gpu.a
	touch cpuall
	$(MAKE)  cpulib -C $(H2O4GPUROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.cpu h2o4gpu-glm-cpu h2o4gpu-glm-cpu-ptr h2o4gpu-kmeans-cpu

h2o4gpu-glm-cpu: Makefile Makefile.cpu $(SOURCESALT) ../../src/build/cpu/h2o4gpu.a
	$(MAKE)  cpulib -C $(H2O4GPUROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.cpu h2o4gpu-glm-cpu

h2o4gpu-glm-cpu-ptr: Makefile Makefile.cpu $(SOURCESPTR)  ../../src/build/cpu/h2o4gpu.a
	$(MAKE)  cpulib -C $(H2O4GPUROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.cpu h2o4gpu-glm-cpu-ptr

h2o4gpu-kmeans-cpu: Makefile Makefile.cpu kmeans_driver.cpp ../../src/build/cpu/h2o4gpu.a
	$(MAKE)  cpulib -C $(H2O4GPUROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.cpu h2o4gpu-kmeans-cpu

# GPU
gpuall: Makefile Makefile.gpu $(SOURCESALT) $(SOURCESPTR) ../../src/build/gpu/h2o4gpu.a
	touch gpuall
	$(MAKE)  gpulib -C $(H2O4GPUROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.gpu h2o4gpu-glm-gpu h2o4gpu-glm-gpu-ptr h2o4gpu-kmeans-gpu

h2o4gpu-glm-gpu: Makefile Makefile.gpu $(SOURCESALT) ../../src/build/gpu/h2o4gpu.a
	$(MAKE)  gpulib -C $(H2O4GPUROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.gpu h2o4gpu-glm-gpu

h2o4gpu-glm-gpu-ptr: Makefile Makefile.gpu $(SOURCESPTR) ../../src/build/gpu/h2o4gpu.a
	$(MAKE)  gpulib -C $(H2O4GPUROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.gpu h2o4gpu-glm-gpu-ptr

h2o4gpu-kmeans-gpu: Makefile Makefile.gpu kmeans_driver.cpp ../../src/build/gpu/h2o4gpu.a
	$(MAKE)  gpulib -C $(H2O4GPUROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.gpu h2o4gpu-kmeans-gpu

run: h2o4gpu-glm-gpu-ptr
	./h2o4gpu-glm-gpu-ptr train.txt 0 $(N) $(N) 100 5 8 1 0 0.2

runcpu: h2o4gpu-glm-cpu-ptr
	./h2o4gpu-glm-cpu-ptr train.txt 0 $(N) $(N) 100 5 8 1 0 0.2

clean:
	rm -f *.o *~ *~
	rm -rf h2o4gpu-glm-cpu h2o4gpu-glm-cpu-ptr h2o4gpu-kmeans-cpu
	rm -rf h2o4gpu-glm-gpu h2o4gpu-glm-gpu-ptr h2o4gpu-kmeans-gpu
	rm -rf *.dSYM
	rm -rf core
