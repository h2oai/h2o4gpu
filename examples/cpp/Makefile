# User Vars
H2OAIGLMROOT=../../src

ifndef N
N=2
endif

#.NOTPARALLEL:

.PHONY: all default clean run runcpu ../../src/build/cpu/h2oaiglm.a ../../src/build/gpu/h2oaiglm.a

all: cpuall gpuall

default: gpuall

SOURCESORIG=lasso.cpp  lasso_path2.cpp  lasso_path.cpp  logistic.cpp  lp_eq.cpp  lp_ineq.cpp  nonneg_l2.cpp run_all_orig.cpp  svm.cpp
SOURCESALT=elastic_net.cpp run_all.cpp
SOURCESPTR=elastic_net_ptr_driver.cpp run_all.cpp

../../src/build/cpu/h2oaiglm.a:
	$(MAKE)  cpulib -C $(H2OAIGLMROOT) IFLAGS=$(IFLAGS)

../../src/build/gpu/h2oaiglm.a:
	$(MAKE)  gpulib -C $(H2OAIGLMROOT) IFLAGS=$(IFLAGS)

# CPU
cpuall: Makefile Makefile.cpu $(SOURCESORIG) $(SOURCESALT) $(SOURCESPTR) ../../src/build/cpu/h2oaiglm.a
	touch cpuall
	$(MAKE)  cpulib -C $(H2OAIGLMROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.cpu h2oai-glm-cpu-orig h2oai-glm-cpu h2oai-glm-cpu-ptr

h2oai-glm-cpu-orig: Makefile Makefile.cpu $(SOURCESORIG) ../../src/build/cpu/h2oaiglm.a
	$(MAKE)  cpulib -C $(H2OAIGLMROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.cpu h2oai-glm-cpu-orig

h2oai-glm-cpu: Makefile Makefile.cpu $(SOURCESALT) ../../src/build/cpu/h2oaiglm.a
	$(MAKE)  cpulib -C $(H2OAIGLMROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.cpu h2oai-glm-cpu

h2oai-glm-cpu-ptr: Makefile Makefile.cpu $(SOURCESPTR)  ../../src/build/cpu/h2oaiglm.a
	$(MAKE)  cpulib -C $(H2OAIGLMROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.cpu h2oai-glm-cpu-ptr

# GPU
gpuall: Makefile Makefile.gpu $(SOURCESORIG) $(SOURCESALT) $(SOURCESPTR) ../../src/build/gpu/h2oaiglm.a
	touch gpuall
	$(MAKE)  gpulib -C $(H2OAIGLMROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.gpu h2oai-glm-gpu-orig h2oai-glm-gpu h2oai-glm-gpu-ptr

h2oai-glm-gpu-orig: Makefile Makefile.gpu $(SOURCESORIG)  ../../src/build/gpu/h2oaiglm.a
	$(MAKE)  gpulib -C $(H2OAIGLMROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.gpu h2oai-glm-gpu-orig

h2oai-glm-gpu: Makefile Makefile.gpu $(SOURCESALT) ../../src/build/gpu/h2oaiglm.a
	$(MAKE)  gpulib -C $(H2OAIGLMROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.gpu h2oai-glm-gpu

h2oai-glm-gpu-ptr: Makefile Makefile.gpu $(SOURCESPTR) ../../src/build/gpu/h2oaiglm.a
	$(MAKE)  gpulib -C $(H2OAIGLMROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -f Makefile.gpu h2oai-glm-gpu-ptr

run: h2oai-glm-gpu-ptr
	./h2oai-glm-gpu-ptr train.txt 0 $(N) $(N) 100 5 5 1 0 0.2

runcpu: h2oai-glm-cpu-ptr
	./h2oai-glm-cpu-ptr train.txt 0 $(N) $(N) 100 5 5 1 0 0.2

clean:
	rm -f *.o *~ *~
	rm -rf h2oai-glm-cpu-orig h2oai-glm-cpu h2oai-glm-cpu-ptr
	rm -rf h2oai-glm-gpu-orig h2oai-glm-gpu h2oai-glm-gpu-ptr
	rm -rf *.dSYM
	rm -rf core
