[H2OAIGLM](https://github.com/h2oai/h2oaiglm)

```text
---

H2OAIGLM is a solver for convex optimization problems in _graph form_ using [Alternating Direction Method of Multipliers] (ADMM).

Requirements
------
CUDA8 for GPU version, OpenMP (for distributed GPU version)


On AWS, upon logging into GPU setup, do at first the below in order to get GPUs to stay warm to avoid delays upon running h2oaiglm.
------

sudo nvidia-smi -pm 1

or

sudo nvidia-persistenced --user foo --persistence-mode # where "foo" is your username


To compile base library:
------

BASE=`pwd`

cd $BASE/src && make -j all

To run gpu C++ version:
------

cd $BASE/examples/cpp && make -j all ; make run

To run 16-gpu version on ipums.txt data:

./h2oai-glm-gpu-ptr ipums.txt 0 16 16 100 6 6 1 0 0.2 &> fold6x6.txt


install R package (assume in h2oaiglm base directory to start with)
------
cd $BASE/src/interface_r && MAKE="make -j" R CMD INSTALL --build h2oaiglm

# Edit interface_r/src/config2.mk and choose TARGET as cpulib or gpulib (currently defaulted to gpulib).


test R package
------
cd $BASE/examples/R && R CMD BATCH simple.R


install python package (assume in h2oaiglm base directory to start with)
-----
cd src/interface_py && python setup.py clean --all && rm -rf h2oaiglm.egg-info && rm -rf h2oaiglm/__pycache__/ && python setup.py install --user

```

Languages / Frameworks
======================
Three different implementations of the solver are either planned or already supported:

  1. C++/BLAS/OpenMP: A CPU version can be found in the file `<h2oaiglm>/src/cpu/`. H2OAIGLM must be linked to a BLAS library (such as the Apple Accelerate Framework or ATLAS).
  2. C++/cuBLAS/CUDA: A GPU version is located in the file `<h2oaiglm>/src/gpu/`. To use the GPU version, the CUDA SDK must be installed, and the computer must have a CUDA-capable GPU.
  3. MATLAB: A MATLAB implementation along with examples can be found in the `<h2oaiglm>/matlab` directory. The code is heavily documented and primarily intended for pedagogical purposes.


Problem Classes
===============

Among others, the solver can be used for the following classes of (linearly constrained) problems

  + Lasso, Ridge Regression, Logistic Regression, Huber Fitting and Elastic Net Regulariation,
  + Total Variation Denoising, Optimal Control,
  + Linear Programs and Quadratic Programs.


References
==========
1. [Parameter Selection and Pre-Conditioning for a Graph Form Solver -- C. Fougner and S. Boyd][h2oaiglm]
2. [Block Splitting for Distributed Optimization -- N. Parikh and S. Boyd][block_splitting]
3. [Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers -- S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein][admm_distr_stats]
4. [Proximal Algorithms -- N. Parikh and S. Boyd][prox_algs]


[pogs]: http://stanford.edu/~boyd/papers/pogs.html "Parameter Selection and Pre-Conditioning for a Graph Form Solver -- C. Fougner and S. Boyd"

[block_splitting]: http://www.stanford.edu/~boyd/papers/block_splitting.html "Block Splitting for Distributed Optimization -- N. Parikh and S. Boyd"

[admm_distr_stats]: http://www.stanford.edu/~boyd/papers/block_splitting.html "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers -- S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein"

[prox_algs]: http://www.stanford.edu/~boyd/papers/prox_algs.html "Proximal Algorithms -- N. Parikh and S. Boyd"


Authors
------
New h2oaiglm: H2O.ai Team
Original Pogs: Chris Fougner, with input from Stephen Boyd. The basic algorithm is from ["Block Splitting for Distributed Optimization -- N. Parikh and S. Boyd"][block_splitting].






