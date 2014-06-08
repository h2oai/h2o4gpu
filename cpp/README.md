### C++ implementation of ADMM for graph form problems.



Tutorial
--------
The fist step is to formulate the problem in such a way that `A`, `f` and `g` are clearly defined. Once this is done, you should instantiate an `PogsData` object and set _all_ input fields. These are:

  + `PogsData::A`: Pointer to the `A` matrix from the problem description. It is assumed to be in row-major format. 
  + `(PogsData::m, PogsData::n)`: Dimensions of `A`.
  + `(PogsData::x, PogsData::y)`: Pointers to pre-allocated memory locations, where the solution will be stored.
  + `(PogsData::f, PogsData::g)`: Vectors of function objects. The `i`'th element corresponds to the term `f_i`  (respectively `g_j`) in the objective. Refer to the Proximal Operator Library section for a description of function objects.


Proximal Operator Library
-------------------------
The heart of the solver is the proximal operator library (`prox_lib.hpp`), which defines proximal operators for a variety of functions. Each function is described by a function object (`FunctionObj`) and a function object is in turn parameterized by five values: `f, a, b, c` and `d`. These correspond to the equation

```
	c * f(a * x - b) + d * x,
```

where `a, b` and `d` take on real values, `c` is a non-negative real and `f` is one of (currently) 13 enumumerated values (see below). To instantiate a `FunctionObj` you must specify all of these values, however `a` and `c` default to 1 and `b` and `d` default to 0. 

The enumerated function types are:

| enum      | Mathematical Function |
| --------- |:----------------------| 
| kAbs      | f(x) = &#124;x&#124;  |
| kHuber    | f(x) = huber(x)       |
| kIdentity | f(x) = x              |  
| kIndBox01 | f(x) = I(0 <= x <= 1) |
| kIndEq0   | f(x) = I(x = 0)       |
| kIndGe0   | f(x) = I(x >= 0)      |
| kIndLe0   | f(x) = I(x <= 0)      |
| kNegLog   | f(x) = -log(x)        |
| kLogistic | f(x) = log(1 + e^x)   |
| kMaxNeg0  | f(x) = max(0, -x)     |
| kMaxPos0  | f(x) = max(0, x)      |
| kSquare   | f(x) = (1/2) x^2      |
| kZero     | f(x) = 0              |

Examples
--------
See `main.cpp` for examples of how to use the solver. We have included these four classes:

  + Non-negative least squares
  + Inequality constrained linear program
  + Equality constrained linear program
  + Support vector machine
