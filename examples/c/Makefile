# User Vars
POGSROOT=../../src

# C Flags
CC=gcc
CFLAGS=$(IFLAGS) -O3 -I$(POGSROOT)/interface_c -Wall -std=c99
LDFLAGS_=-lstdc++ -lm

# CUDA Flags
CULDFLAGS_=-lstdc++ -lcudart -lcublas -lcusparse

# Check System Args.
UNAME = $(shell uname -s)
ifeq ($(UNAME), Darwin)
LDFLAGS=$(LDFLAGS_) -framework Accelerate
CULDFLAGS=-lm -L/usr/local/cuda/lib $(CULDFLAGS_)
else
LDFLAGS=$(LDFLAGS_) -lopenblas
CULDFLAGS=-lm -L/usr/local/cuda/lib64 $(CULDFLAGS_)
endif

# CPU
cpu: lasso.c
	$(MAKE) cpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -C $(POGSROOT)/interface_c pogs_c.o
	$(CC) $(CFLAGS) -o run $< $(POGSROOT)/interface_c/pogs_c.o \
		$(POGSROOT)/build/pogs.a $(LDFLAGS)

# GPU
gpu: lasso.c
	$(MAKE) gpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(MAKE) -C $(POGSROOT)/interface_c pogs_c.o
	$(CC) $(CFLAGS) -o run $< $(POGSROOT)/interface_c/pogs_c.o \
		$(POGSROOT)/build/pogs.a $(CULDFLAGS)

clean:
	rm -f *.o *~ *~ run 
	rm -rf *.dSYM

