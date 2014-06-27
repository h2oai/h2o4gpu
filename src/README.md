## C++ Implementation of POGS



Tutorial
--------
The fist step is to formulate the problem in such a way that `A`, `f` and `g` are clearly defined. Once this is done, you should instantiate an `PogsData` object and set _all_ input fields. These are:

  + `PogsData::A`: Pointer to the `A` matrix from the problem description. It is assumed to be in row-major format. 
  + `(PogsData::m, PogsData::n)`: Dimensions of `A`.
  + `(PogsData::x, PogsData::y)`: Pointers to pre-allocated memory locations, where the solution will be stored.
  + `(PogsData::f, PogsData::g)`: Vectors of function objects. The `i`'th element corresponds to the term `f_i`  (respectively `g_i`) in the objective. Refer to the Proximal Operator Library section for a description of function objects.


Proximal Operator Library
-------------------------
The heart of the solver is the proximal operator library (`prox_lib.hpp`), which defines proximal operators for a variety of functions. Each function is described by a function object (`FunctionObj`) and a function object is in turn parameterized by five values: `h, a, b, c` and `d`. These correspond to the equation

```
	c * h(a * x - b) + d * x,
```

where `a, b` and `d` take on real values, `c` is a non-negative real and `h` is one of (currently) 16 enumumerated values (see below). To instantiate a `FunctionObj` you must specify all of these values, however `a` and `c` default to 1 and `b` and `d` default to 0. 

The enumerated function types are:

| enum      | Mathematical Function |
| --------- |:----------------------| 
| kAbs      | h(x) = &#124;x&#124;  |
| kEntr     | h(x) = x log(x)       |
| kExp      | h(x) = exp(x)         |
| kHuber    | h(x) = huber(x)       |
| kIdentity | h(x) = x              |  
| kIndBox01 | h(x) = I(0 <= x <= 1) |
| kIndEq0   | h(x) = I(x = 0)       |
| kIndGe0   | h(x) = I(x >= 0)      |
| kIndLe0   | h(x) = I(x <= 0)      |
| kLogistic | h(x) = log(1 + e^x)   |
| kMaxNeg0  | h(x) = max(0, -x)     |
| kMaxPos0  | h(x) = max(0, x)      |
| kNegLog   | h(x) = -log(x)        |
| kSquare   | h(x) = (1/2) x^2      |
| kRecipr   | h(x) = 1/x            |
| kZero     | h(x) = 0              |

Examples
--------
See `test.cpp` for examples of how to use the solver. We have included five classes of problems:

  + Non-negative least squares
  + Inequality constrained linear program
  + Equality constrained linear program
  + Support vector machine
  + Lasso
