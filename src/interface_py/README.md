## Python Interface for POGS
About
-----


Requirements
------------
The POGS c++/CUDA has the following dependencies:

	Linux with g++ >= 4.8 ~or~ OSX with clang >= 3.3 (for C++11 compatibility)
	CUDA 	>= 6.0 (for GPU solver). 
	gcc

The python package pogs has the following additional dependencies:

	python2 >= 2.7
	numpy	>= 1.8
	scipy	>= 0.13


Installation
------------

If would like to compile POGS for GPU, please make sure your $PATH environment variable contains the path to the CUDA binaries (namely nvcc)


```
> export PATH=$PATH:<path-to-cuda-bin>
```

In the shell, execute:

```
> cd("<path-to-pogs-root>/src/interface_py/")
```

If desired, create a virtual environment before the installation step. 


```
> sudo PATH=$PATH python setup.py install
```

The python installer will build the POGS source if needed and register this package with your Python distribution as pogs. 


Tutorial
--------

##POGS Functions
pogs.

##Solver

### Initalize

s=pogs.SolverCPU(A)

s=pogs.SolverGPU(A)

s.pysolution
s.info

### Solve
s.solve(f,g,**kwargs)

The keyword arguments are

	max_iter:		int, default = 
	gap_stop:     	bool, 
	rel_tol: 		floating, default = 1e-3
	abs_tol:		float, default = 1e-3
	rho:			float, default = 1 or last value
	adaptive_rho:	bool, default = True
	warm start:		bool, default = False
	x_init:			numpy ndarray float32/64, initial guess for primal variable x (warm start)
	nu_init:		numpy ndarray float32/64, initial guess for dual variable nu (warm start)



### Solver Status
pogs.STATUS[s.info]



### Shut down
To release memory allocated in the C++/CUDA instance of the solver, call

```
test comment?
>s.finish()
```


Examples
--------
See the `<pogs>/examples/py/` directory for examples of how to use the solver. We have included six classes of problems:

  + Non-negative least squares
  + Inequality constrained linear program
  + Equality constrained linear program
  + Support vector machine
  + Lasso
  + Logistic regression
