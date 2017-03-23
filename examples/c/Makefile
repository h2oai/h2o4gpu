# User Vars
POGSROOT=../../src

# C Flags
CC=gcc
CFLAGS=$(IFLAGS) -O3 -I$(POGSROOT)/interface_c -Wall -std=c99
LDFLAGS_=-lstdc++ -lm -fopenmp

# CUDA Flags
CULDFLAGS_=-lstdc++ -lcudart -lcublas -lcusparse

# Check System Args.
ifeq ($(UNAME), Darwin)
LDFLAGS=$(LDFLAGS_) -framework Accelerate
CULDFLAGS=-lm -L/usr/local/cuda/lib64 $(CULDFLAGS_)
else
LDFLAGS=$(LDFLAGS_) -L/home/jon/lib -L/usr/local/cuda/lib64     -L/usr/lib/   -lmagma -lcublas -lcudart -lopenblas #-lopenblas
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

