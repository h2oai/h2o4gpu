[POGS](https://github.com/h2oai/pogs)

Based upon https://github.com/foges/pogs
====

Proximal Graph Solver (POGS) is a solver for convex optimization problems in _graph form_ using [Alternating Direction Method of Multipliers](http://foges.github.io/pogs/ref/admm) (ADMM). 

Requirements
------
R, CUDA8 for GPU version

On AWS, upon logging into GPU setup, do at first the below in order to get GPUs to stay warm to avoid delays upon running pogs.
------

sudo nvidia-smi -pm 1


To compile gpu version:
------

cd src ; make ; cd ../examples/cpp ; make

To run gpu version if one has 16 gpus (e.g. on AWS), where first and 3rd argument should be same for any number of GPUs.  Dataset is currently small called simple.txt, but ask Arno or Jon for the larger census-based data set to highlight 100% multiple-GPU usage.
------


./h2oai-glm-gpu 16 100 16 0.2



install R package (assume in pogs base directory to start with)
------
cd src/interface_r
Edit interface_r/src/config.mk and choose TARGET as cpu or gpu (currently defaulted to gpu).
R CMD INSTALL --build pogs

test R package
------
cd ../../
cd examples/R
R CMD BATCH simple.R




Details
-----------------------------------

- The GitHub.io page contains [Documentation](http://foges.github.io/pogs).
- A [paper](http://stanford.edu/~boyd/papers/pogs.html) online, with details about the implementation, as well as many numerical examples.


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

![Supported Equations](https://github.com/foges/pogs/raw/master/img/eqs.png)

where `I(.)` is the indicator function, taking on the value 0 if the condition is satisfied and infinity otherwise. More functions can be added by modifying the proximal operator header file: `<pogs>/src/include/prox_lib.h`.


Languages / Frameworks
======================
Three different implementations of the solver are either planned or already supported:

  1. C++/BLAS/OpenMP: A CPU version can be found in the file `<pogs>/src/cpu/`. POGS must be linked to a BLAS library (such as the Apple Accelerate Framework or ATLAS).
  2. C++/cuBLAS/CUDA: A GPU version is located in the file `<pogs>/src/gpu/`. To use the GPU version, the CUDA SDK must be installed, and the computer must have a CUDA-capable GPU.
  3. MATLAB: A MATLAB implementation along with examples can be found in the `<pogs>/matlab` directory. The code is heavily documented and primarily intended for pedagogical purposes.


Problem Classes
===============

Among others, the solver can be used for the following classes of (linearly constrained) problems

  + Lasso, Ridge Regression, Logistic Regression, Huber Fitting and Elastic Net Regulariation,
  + Total Variation Denoising, Optimal Control,
  + Linear Programs and Quadratic Programs.


References
==========
1. [Parameter Selection and Pre-Conditioning for a Graph Form Solver -- C. Fougner and S. Boyd][pogs]
2. [Block Splitting for Distributed Optimization -- N. Parikh and S. Boyd][block_splitting]
3. [Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers -- S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein][admm_distr_stats]
4. [Proximal Algorithms -- N. Parikh and S. Boyd][prox_algs]


[pogs]: http://stanford.edu/~boyd/papers/pogs.html "Parameter Selection and Pre-Conditioning for a Graph Form Solver -- C. Fougner and S. Boyd"

[block_splitting]: http://www.stanford.edu/~boyd/papers/block_splitting.html "Block Splitting for Distributed Optimization -- N. Parikh and S. Boyd"

[admm_distr_stats]: http://www.stanford.edu/~boyd/papers/block_splitting.html "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers -- S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein"

[prox_algs]: http://www.stanford.edu/~boyd/papers/prox_algs.html "Proximal Algorithms -- N. Parikh and S. Boyd"


Authors
------
Chris Fougner, with input from Stephen Boyd. The basic algorithm is from ["Block Splitting for Distributed Optimization -- N. Parikh and S. Boyd"][block_splitting].






