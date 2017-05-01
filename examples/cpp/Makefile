# User Vars
H2OAIGLMROOT=../../src

ifndef N
N=2
endif

.NOTPARALLEL:

.PHONY: all default cpuall gpuall run runcpu clean


all: cpuall gpuall

default: gpuall

# CPU
cpuall: Makefile.cpu
	$(MAKE)  cpu -C $(H2OAIGLMROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.cpu h2oai-glm-cpu-orig h2oai-glm-cpu h2oai-glm-cpu-ptr

h2oai-glm-cpu-orig: Makefile.cpu
	$(MAKE)  cpu -C $(H2OAIGLMROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.cpu h2oai-glm-cpu-orig

h2oai-glm-cpu: Makefile.cpu
	$(MAKE)  cpu -C $(H2OAIGLMROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.cpu h2oai-glm-cpu

h2oai-glm-cpu-ptr: Makefile.cpu
	$(MAKE)  cpu -C $(H2OAIGLMROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.cpu h2oai-glm-cpu-ptr

# GPU
gpuall: Makefile.gpu
	$(MAKE)  gpu -C $(H2OAIGLMROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.gpu h2oai-glm-gpu-orig h2oai-glm-gpu h2oai-glm-gpu-ptr

h2oai-glm-gpu-orig: Makefile.gpu
	$(MAKE)  gpu -C $(H2OAIGLMROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.gpu h2oai-glm-gpu-orig

h2oai-glm-gpu: Makefile.gpu
	$(MAKE)  gpu -C $(H2OAIGLMROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.gpu h2oai-glm-gpu

h2oai-glm-gpu-ptr: Makefile.gpu
	$(MAKE)  gpu -C $(H2OAIGLMROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.gpu h2oai-glm-gpu-ptr

run: h2oai-glm-gpu-ptr
	./h2oai-glm-gpu-ptr train.txt 0 $(N) $(N) 100 16 1 0 0.2

runcpu: h2oai-glm-cpu-ptr
	./h2oai-glm-cpu-ptr train.txt 0 $(N) $(N) 100 16 1 0 0.2

clean:
	rm -f *.o *~ *~
	rm -rf h2oai-glm-cpu-orig h2oai-glm-cpu h2oai-glm-cpu-ptr
	rm -rf h2oai-glm-gpu-orig h2oai-glm-gpu h2oai-glm-gpu-ptr
	rm -rf *.dSYM
	rm -rf core
