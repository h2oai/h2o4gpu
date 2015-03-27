## C++ Implementation of POGS

Tutorial
--------
Please refer to the examples to see how to use POGS. Documentation will be updated in the near future.


Proximal Operator Library
-------------------------
The heart of the solver is the proximal operator library (`prox_lib.h`), which defines proximal operators for a variety of functions. Each function is described by a function object (`FunctionObj`) and a function object is in turn parameterized by five values: `h, a, b, c` and `d`. These correspond to the equation

```
	c * h(a * x - b) + d * x + e * x ^ 2,
```

where `a, b` and `d` take on real values, `c, d` are non-negative real and `h` is one of (currently) 16 enumumerated values (see below). To instantiate a `FunctionObj` you must specify all of these values, however `a` and `c` default to 1 and `b` and `d` default to 0. 

The enumerated function types are:

| enum      | Mathematical Function | Domain  |
| --------- |:----------------------|:--------|
| kAbs      | h(x) = &#124;x&#124;  |R        |
| kNegEntr  | h(x) = x log(x)       |[0, inf) |
| kExp      | h(x) = exp(x)         |R        |
| kHuber    | h(x) = huber(x)       |R        |
| kIdentity | h(x) = x              |R        |
| kIndBox01 | h(x) = I(0 <= x <= 1) |[0, 1]   |
| kIndEq0   | h(x) = I(x = 0)       |{0}      |
| kIndGe0   | h(x) = I(x >= 0)      |[0, inf) |
| kIndLe0   | h(x) = I(x <= 0)      |(-inf, 0]|
| kLogistic | h(x) = log(1 + e^x)   |R        |
| kMaxNeg0  | h(x) = max(0, -x)     |R        |
| kMaxPos0  | h(x) = max(0, x)      |R        |
| kNegLog   | h(x) = -log(x)        |(0, inf) |
| kSquare   | h(x) = (1/2) x^2      |R        |
| kRecipr   | h(x) = 1/x            |(0, inf) |
| kZero     | h(x) = 0              |R        |

The function `I(.)` is the indicator function, taking on the value 0 if the condition is satisfied and infinity otherwise. Notice that some functions have an implicitly constrained domain.

Examples
--------
See the `<pogs>/examples/cpp/` directory for examples of how to use the solver. We have included six classes of problems:

  + Non-negative least squares
  + Inequality constrained linear program
  + Equality constrained linear program
  + Support vector machine
  + Lasso
  + Logistic regression
