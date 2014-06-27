## MATLAB interface to POGS

The MATLAB interface is a single MEX-function, with the signature 

```
[x, y, optval] = pogs(A, f, g, params)
```

where `A` is a matrix, `f`, `g`, and  `params` are structs, `x` and `y` are vectors, and `optval` is a scalar.  The structs `f` and `g` have fields `h`, `a`, `b`, `c` and `d`, each of which must either be a vector of dimension `size(A, 1)` (resp. `size(A, 2)`) or a scalar. If a scalar is specified, then it is assumed that the scalar should be repeated `size(A,1)` (resp. `size(A,2)`) times. All fields except `h` are optional. The `params` struct has fields `rel_tol, abs_tol, rho, max_iter` and `quiet`. Specifying `params` is optional.

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
kAbs      = 0; kEntr     = 1; kExp     = 2; kHuber = 3;   kIdentity = 4; 
kIndBox01 = 5; kIndEq0   = 6; kIndGe0  = 7; kIndLe0 = 8;  kLogistic = 9;
kMaxNeg0 = 10; kMaxPos0 = 11; kNegLog = 12; kRecipr = 13; kSquare  = 14;
kZero    = 15;
```

Compiling
---------
To compile wrapper type

```
pogs_setup
```

in the MATLAB console.  



