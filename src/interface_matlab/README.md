## MATLAB interface to POGS

The MATLAB interface is a single MEX-function, with the signature 

```
[x, y, optval] = pogs(A, f, g, params)
```

where `A` is a matrix, `f`, `g`, and  `params` are structs, `x` and `y` are vectors, and `optval` is a scalar.  The structs `f` and `g` have fields `h`, `a`, `b`, `c`, `d` and `e`, each of which must either be a vector of dimension `size(A, 1)` (resp. `size(A, 2)`) or a scalar. If a scalar is specified, then it is assumed that the scalar should be repeated `size(A,1)` (resp. `size(A,2)`) times. All fields except `h` are optional. The `params` struct has fields `rel_tol, abs_tol, rho, max_iter` and `quiet`. Specifying `params` is optional.

Example
-------

The Lasso problem

```
min. (1/2)||A x - b||_2^2 + \lambda ||x||_1
```
can be specified using these five lines

```
f.h = kSquare;
f.b = b;
g.h = kAbs;
g.c = lambda;
[x, y, optval] = pogs(A, f, g);
```
Both `kSquare` and `kAbs` correspond to one of 16 numeric values:

```
kAbs      = 0; kExp      = 1; kHuber   = 2; kIdentity = 3; kIndBox01 = 4;
kIndEq0   = 5; kIndGe0   = 6; kIndLe0  = 7; kLogistic = 8; kMaxNeg0  = 9;
kMaxPos0 = 10; kNegEntr = 11; kNegLog = 12; kRecipr  = 13; kSquare  = 14;
kZero    = 15;
```

Compiling
---------
To compile wrapper type

```
pogs_setup
```

in the MATLAB console.  



