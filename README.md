POGS
====

POGS is a solver for convex optimization problems in _graph form_ using [Alternating Direction Method of Multipliers][admm_distr_stats] (ADMM). The algorithm is described in [this paper][block_splitting]. 

----
A graph form problem can be expressed as

```
minimize        f(y) + g(x)
subject to      y = A * x
```
where `f` and `g` are convex and can take on the values `R \cup {∞}`. The solver requires that the [proximal operators][prox_algs] of `f` and `g` are known and that `f` and `g` are separable, meaning that they can be written as

```
f(y) = \sum_i f_i(y_i)
g(x) = \sum_i g_i(x_i)
```

The following functions are currently supported

  + `f(x) = |x|`
  + `f(x) = x log(x)`
  + `f(x) = e^x`
  + `f(x) = huber(x)`
  + `f(x) = x`
  + `f(x) = I(0 <= x <= 1)`
  + `f(x) = I(x = 0)`
  + `f(x) = I(x >= 0)`
  + `f(x) = I(x <= 0)`
  + `f(x) = log(1 + e^x)`
  + `f(x) = max(0, -x)`
  + `f(x) = max(0, x)`
  + `f(x) = -log(x)`
  + `f(x) = 1/x`
  + `f(x) = (1/2) x^2`
  + `f(x) = 0`
  
More functions can be added by modifying the proximal operator header file: `<pogs>/cpp/prox_lib.hpp`.

Languages / Frameworks
======================
Three different implementations of the solver are either planned or already supported:

  1. MATLAB: A MATLAB implementation along with examples can be found in the `<pogs>/matlab` directory.
  2. C++/BLAS/GSL/OpenMP: A multithreaded `C++` version can be found in the `<pogs>/cpp` directory. For best performance it's important to link to  a tuned BLAS or LAPACK library, such as ATLAS or Accelerate Framework.
  3. C++/cuBLAS/CUDA: A GPU implementation is planned.


Problem Classes
===============

Among others, the solver can be used for the following classes of (linearly constrained) problems

  + Least Squares
  + Lasso, Ridge Regression, Logistic Regression, Huber Fitting and Elastic Net Regulariation 
  + Linear Programs and Quadratic Programs
  + Analytic Centering


References
==========
1. [Block Splitting for Distributed Optimization -- N. Parikh and S. Boyd][block_splitting]
2. [Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers -- S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein][admm_distr_stats]
3. [Proximal Algorithms -- N. Parikh and S. Boyd][prox_algs]


[block_splitting]: http://www.stanford.edu/~boyd/papers/block_splitting.html "Block Splitting for Distributed Optimization -- N. Parikh and S. Boyd"

[admm_distr_stats]: http://www.stanford.edu/~boyd/papers/block_splitting.html "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers -- S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein"

[prox_algs]: http://www.stanford.edu/~boyd/papers/prox_algs.html "Proximal Algorithms -- N. Parikh and S. Boyd"


Author
------
Chris Fougner (fougner@stanford.edu)

Acknowledgement: Much of this is based on work by Neal Parikh.

