# Python interface for H2O4GPU

About
-----


Requirements
------------
The H2O4GPU c++/CUDA library has the following dependencies:

	g++ >= 4.8 (C_++11 compatibility, Linux)
	clang >= 3.3 (C_++11 compatibility, OSX)
	CUDA >= 6.0 (for GPU solver)

The Python package h2o4gpu has the following additional dependencies:

	python 3.6
	numpy >= 1.8
	scipy >= 0.13


Installation
------------

If would like to compile H2O4GPU for GPU, please make sure your `$PATH` environment variable contains the path to the CUDA binaries (namely `nvcc`)


```bash
$ export PATH=$PATH:<path-to-cuda-bin>
```

In the shell, navigate to the directory:

```bash
$ cd("<path-to-h2o4gpu-root>/src/interface_py/")
```

If desired, create a virtual environment before the installation step (https://virtualenv.pypa.io/en/latest/).

```bash
$ [sudo] PATH=$PATH python setup.py install
```

The python installer will build the H2O4GPU source if needed and register this package with your Python distribution as h2o4gpu.


Usage
-----

After installing, import the package `h2o4gpu` to use it in a script.

```python
> import h2o4gpu as h2o4gpu
```

The import should print a couple success messages:
```python

> Loaded H2O4GPU CPU Library
>
> Loaded H2O4GPU GPU Library
```

If either of the cpu/gpu shared object libraries (wrapped by this interface) cannot be located, a warning will be printed instead.


###<u>H2O4GPU functions

The Python wrapper for H2O4GPU works for fully separable obective functions, i.e., those that can be written as


	f(y) = sum_{i=1}^m f_i(y_i)

In particular, H2O4GPU specifies the format of these elementwise functions as:

	f_i(y_i)=c_i * h_i(a_i * y_i - b_i) + d_i y)i + e_i y_i^2,

where `a_i`, `b_i`, `c_i`, `d_i`, and `e_i` are parameters, and `h_i(.)` must be one of several enumerated functions. The dictionary
```python
> h2o4gpu.FUNCTION
```
enumerates the C/C++ `enum` integer codes corresponding to these enumerated functions. They are given as follows:

Function key | Function code | Function key | Function Code
-------------|---------------|--------------|--------------
`'ABS'`		| 0 			 | `'LOGISTIC'` | 8
`'EXP'`		| 1 			 | `'MAXNEG0'` 	| 9
`'HUBER'`	| 2				 | `'MAXPOS0'` 	| 10
`'IDENTITY'`| 3				 | `'NEGENTR'` 	| 11
`'INDBOX01'`| 4				 | `'NEGLOG'` 	| 12
`'INDEQ0'`	| 5				 | `'RECIPR'` 	| 13
`'INDGE0'`	| 6				 | `'SQUARE'` 	| 14
`'INDLE0'`	| 7				 | `'ZERO'` 	| 15


To build a H2O4GPU function vector, call:
```python
> f=h2o4gpu.FunctionVector(m,double_precision=False)
```
where `m` is an integer specifying the vector length and the keyword argument `double_precision` (defaulting to `False`) sets the data type for the FunctionVector object.

Each of the fields
`f.a`, `f.b`, `f.c`, `f.d`, and `f.e`
is a numpy `ndarray` of floating point values of length `m`, where the entries of these vectors specify the parameters of the elementwise function `f_i(.)` as described above.

The field `f.h` is a numpy `ndarray` of length `m` and its entries must be integers corresponding to valid H2O4GPU function codes, (those specificied in the table above). For instance, we may set all the entries function vector `f` to be least-square losses by calling:

```python
> f.h[:]=h2o4gpu.FUNCTION['SQUARE']
```

###<u> Solver initialization

To initialize the solver, call

```python
> s=h2o4gpu.SolverCPU(A)
```
or
```python
> s=h2o4gpu.SolverGPU(A)
```

The matrix `A` should be a numpy `ndarray` (dense) or scipy `csr_matrix` or `csc_matrix` (sparse) containing single or double precision floating point values.


###<u> Solver calls

To run the solver, call

```python
> s.solve(f,g,**kwargs)
```

The inputs `f` and `g` must be of type `h2o4gpu.FunctionVector` with associated lengths `m` and `n`, respectively, for a solver initialized with an `m` x `n` matrix. The floating point precision of the `FunctionVector` objects must match that of the solver's associated matrix.

The keyword arguments are:

*solver settings*
+ `max_iters`: integer, default = 2500
+ `rel_tol`: floating, default = 1e-3
+ `abs_tol`: float, default = 1e-3
+ `gap_stop`: boolean, default = True
+ `rho`:			float, default = 1 (first run) or  final value from previous run
+ `adaptive_rho`:	boolean, default = True
+ `equil`:          boolean, default = True
+ `warm start`:		boolean, default = False
+ `nDev`:		    int, default = 1
+ `wDev`:		    int, default = 1

*solver intialization*
+ `x_init`: `numpy.ndarray` (float32/64), initial guess for primal variable x (warm start)
+ `nu_init`: `numpy.ndarray` (float32/64), initial guess for dual variable nu (warm start)


The solver settings are persistent except for `warm_start`. Thus, all settings will remain as set (e.g., at their defaults) until explicitly changed by the user.

Since `adaptive_rho` is enabled by default, the last value of `rho` from the previous solver run is propagated unless overridden by a new value from the user.

After the first run, the C++ solver will automatically resume each subsequent `solve()` call from the optimal point attained at the end of the previous solver run. Therefore, `warm_start=True` is to be used only when overriding the default warm start behavior, i.e., when supplying guesses for x (via `x_init`) and/or nu (via `nu_init`).


###<u> Solver status & results

The solver object has a field `info` containing the following information:

```python
> s.info.obj 	# objective value
> s.info.iters 	# number of iterations performed
> s.info.solvetime 	# run time
> s.info.status 	# solver status code
> s.info.rho 	#  value of rho on final iteration
```

To get the text version of the status code, call:
```python
> h2o4gpu.STATUS[s.info.status]
```

The post-run state of the solver variables can be accessed at:

```python
#primal variables
> s.solution.x
> s.solution.y
#dual variables
> s.solution.mu
> s.solution.nu
```

###<u> Freeing the solver
To release memory allocated in the C++/CUDA instance of the solver, call:

```python
> s.finish()
```

Examples
--------
See the `<path-to-h2o4gpu-root>/examples/py/` directory for examples of how to use the solver. We have included six classes of problems:

+ Non-negative least squares
+ Inequality constrained linear program
+ Equality constrained linear program
+ Support vector machine
+ Lasso
+ Logistic regression
