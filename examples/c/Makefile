# User Vars
POGSROOT=../../src

# C Flags
CC=gcc
CFLAGS=-O3 -I$(POGSROOT)/interface_c -Wall -std=c99
LDFLAGS_=-lstdc++ -lm

# CUDA Flags
CULDFLAGS_=-lstdc++ -lcudart -lcublas -lcusparse

# Check System Args.
UNAME = $(shell uname -s)
ifeq ($(UNAME), Darwin)
LDFLAGS=$(LDFLAGS_) -framework Accelerate
CULDFLAGS=-L/usr/local/cuda/lib $(CULDFLAGS_)
else
LDFLAGS=$(LDFLAGS_)
CULDFLAGS=-L/usr/local/cuda/lib64 $(CULDFLAGS_)
endif

# CPU
cpu: lasso.c
	$(MAKE) -C $(POGSROOT) pogs.o
	$(MAKE) -C $(POGSROOT)/interface_c pogs_c.o
	$(CC) $(CFLAGS) -o lasso $< \
		$(POGSROOT)/pogs.o $(POGSROOT)/interface_c/pogs_c.o $(LDFLAGS)

# GPU
gpu: lasso.c
	$(MAKE) -C $(POGSROOT) pogs_cu.o pogs_sp_cu.o pogs_cu_link.o
	$(MAKE) -C $(POGSROOT)/interface_c pogs_c.o
	$(CC) $(CFLAGS) -o lasso $< \
		$(POGSROOT)/pogs_sp_cu.o $(POGSROOT)/pogs_cu.o $(POGSROOT)/pogs_cu_link.o \
		$(POGSROOT)/cml.o $(POGSROOT)/interface_c/pogs_c.o $(CULDFLAGS)

clean:
	rm -f *.o *~ *~ lasso 
	rm -rf *.dSYM

