[POGS](http://foges.github.io/pogs)
====

POGS is a solver for convex optimization problems in _graph form_ using [Alternating Direction Method of Multipliers](http://foges.github.io/pogs/ref/admm) (ADMM). Head over to the GitHub.io page for complete [Documentation](http://foges.github.io/pogs).

----
A graph form problem can be expressed as

```
minimize        f(y) + g(x)
subject to      y = Ax
```
where `f` and `g` are convex and can take on the values `R \cup {∞}`. The solver requires that the [proximal operators](http://foges.github.io/pogs/ref/admm) of `f` and `g` are known and that `f` and `g` are separable, meaning that they can be written as

```
f(y) = sum_{i=1}^m f_i(y_i)
g(x) = sum_{i=1}^n g_i(x_i)
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

where `I(.)` is the indicator function, taking on the value 0 if the condition is satisfied and infinity otherwise. More functions can be added by modifying the proximal operator header file: `<pogs>/src/prox_lib.h`.


Languages / Frameworks
======================
Three different implementations of the solver are either planned or already supported:

  1. C++/BLAS/OpenMP: A CPU version can be found in the file `<pogs>/src/pogs.cpp`. POGS must be linked to a BLAS library (such as the Apple Accelerate Framework or ATLAS).
  2. C++/cuBLAS/CUDA: A GPU version is located in the file `<pogs>/src/pogs.cu`. To use the GPU version, the CUDA SDK must be installed, and the computer must have a CUDA-capable GPU.
  3. MATLAB: A MATLAB implementation along with examples can be found in the `<pogs>/matlab` directory. The code is heavily documented and primarily intended for pedagogical purposes.


Problem Classes
===============

Among others, the solver can be used for the following classes of (linearly constrained) problems

  + Lasso, Ridge Regression, Logistic Regression, Huber Fitting and Elastic Net Regulariation,
  + Total Variation Denoising, Optimal Control,
  + Linear Programs and Quadratic Programs.


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

Acknowledgement: POGS is partially based on work by Neal Parikh. In particular the term _graph form_ and many of the derivations are taken from ["Block Splitting for Distributed Optimization -- N. Parikh and S. Boyd"][block_splitting].

