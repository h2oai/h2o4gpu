# User Vars
POGSROOT=../../src

.NOTPARALLEL:

all: cpuall gpuall

default: gpuall

# CPU
cpuall: Makefile.cpu
	$(MAKE)  cpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.cpu cpu cpualt cpumapd

cpu: Makefile.cpu
	$(MAKE)  cpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.cpu cpu

cpualt: Makefile.cpu
	$(MAKE)  cpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.cpu cpualt

cpumapd: Makefile.cpu
	$(MAKE)  cpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.cpu cpumapd

# GPU
gpuall: Makefile.gpu
	$(MAKE)  gpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.gpu gpu gpualt gpumapd

gpu: Makefile.gpu
	$(MAKE)  gpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.gpu gpu

gpualt: Makefile.gpu
	$(MAKE)  gpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.gpu gpualt

gpumapd: Makefile.gpu
	$(MAKE)  gpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.gpu gpumapd

run: gpumapd
	./h2oai-glm-gpu-mapd 1 100 16 1 0 0.2

runcpu: cpumapd
	./h2oai-glm-cpu-mapd 1 100 1 1 0 0

clean:
	rm -f *.o *~ *~
	rm -rf runcpu h2oai-glm-cpu h2oai-glm-cpu-mapd
	rm -rf rungpu h2oai-glm-gpu h2oai-glm-gpu-mapd
	rm -rf *.dSYM
	rm -rf core


