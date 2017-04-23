# User Vars
POGSROOT=../../src

.NOTPARALLEL:

all: cpuall gpuall

default: gpuall

# CPU
cpuall: Makefile.cpu
	$(MAKE)  cpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.cpu cpu cpualt cpuptr

cpu: Makefile.cpu
	$(MAKE)  cpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.cpu cpu

cpualt: Makefile.cpu
	$(MAKE)  cpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.cpu cpualt

cpuptr: Makefile.cpu
	$(MAKE)  cpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.cpu cpuptr

# GPU
gpuall: Makefile.gpu
	$(MAKE)  gpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.gpu gpu gpualt gpuptr

gpu: Makefile.gpu
	$(MAKE)  gpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.gpu gpu

gpualt: Makefile.gpu
	$(MAKE)  gpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.gpu gpualt

gpuptr: Makefile.gpu
	$(MAKE)  gpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.gpu gpuptr

run: gpuptr
	./h2oai-glm-gpu-ptr 2 2 100 16 1 0 0.2

runcpu: cpuptr
	./h2oai-glm-cpu-ptr 1 1 100 16 1 0 0.2

clean:
	rm -f *.o *~ *~
	rm -rf runcpu h2oai-glm-cpu h2oai-glm-cpu-ptr
	rm -rf rungpu h2oai-glm-gpu h2oai-glm-gpu-ptr
	rm -rf *.dSYM
	rm -rf core


