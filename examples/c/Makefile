# User Vars
H2OAIGLMROOT=../../src

# C Flags
CC=gcc
CFLAGS=$(IFLAGS) -O3 -I$(H2OAIGLMROOT)/interface_c -Wall -std=c99
LDFLAGS_=-lstdc++ -lm

# CUDA Flags
CULDFLAGS_=-lstdc++ -lcudart -lcublas -lcusparse

# Check System Args.
UNAME = $(shell uname -s)
ifeq ($(UNAME), Darwin)
LDFLAGS=$(LDFLAGS_) -framework Accelerate
CULDFLAGS=-lm -L/usr/local/cuda/lib $(CULDFLAGS_)
else
LDFLAGS=$(LDFLAGS_) -lblas
CULDFLAGS=-lm -L/usr/local/cuda/lib64 $(CULDFLAGS_)
endif

# CPU
cpu: lasso.c
	$(MAKE) cpu -C $(H2OAIGLMROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -C $(H2OAIGLMROOT)/interface_c h2oaiglm_c.o
	$(CC) $(CFLAGS) -o run $< $(H2OAIGLMROOT)/interface_c/h2oaiglm_c.o \
		$(H2OAIGLMROOT)/build/h2oaiglm.a $(LDFLAGS)

# GPU
gpu: lasso.c
	$(MAKE) gpu -C $(H2OAIGLMROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -C $(H2OAIGLMROOT)/interface_c h2oaiglm_c.o
	$(CC) $(CFLAGS) -o run $< $(H2OAIGLMROOT)/interface_c/h2oaiglm_c.o \
		$(H2OAIGLMROOT)/build/h2oaiglm.a $(CULDFLAGS)

clean:
	rm -f *.o *~ *~ run 
	rm -rf *.dSYM

