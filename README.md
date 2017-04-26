[H2OAIGLM](https://github.com/h2oai/h2oaiglm)

---

H2OAIGLM is a solver for convex optimization problems in _graph form_ using [Alternating Direction Method of Multipliers] (ADMM).

Requirements
------
R, CUDA8 for GPU version, OpenMP (unless remove -fopenmp from all Makefile's in all directories).

On AWS, upon logging into GPU setup, do at first the below in order to get GPUs to stay warm to avoid delays upon running h2oaiglm.
------

sudo nvidia-smi -pm 1

or

sudo nvidia-persistenced --user foo --persistence-mode # where "foo" is your username


To compile gpu version:
------

cd src ; make -j ; cd ../examples/cpp ; make -j gpuall

To run gpu version if one has 16 gpus (e.g. on AWS), where first and 3rd argument should be same for any number of GPUs.  Dataset is currently small called simple.txt, but ask Arno or Jon for the larger census-based data set to highlight 100% multiple-GPU usage.
------

./h2oai-glm-gpu 16 100 16 1 0 0.2

To compile cpu version:
------

cd src ; make -j cpu ; cd ../examples/cpp ; make -j cpuall

For otherwise identical CPU run on all CPU's cores:

./h2oai-glm-cpu 1 100 16 1 0 0.2


install R package (assume in h2oaiglm base directory to start with)
------
cd src/interface_r
Edit interface_r/src/config.mk and choose TARGET as cpu or gpu (currently defaulted to gpu).
MAKE="make -j" R CMD INSTALL --build h2oaiglm

test R package
------
cd ../../
cd examples/R
R CMD BATCH simple.R


install python package (assume in h2oaiglm base directory to start with)
-----
cd src/interface_py
python setup.py clean --all
rm -rf h2oaiglm.egg-info
rm -rf h2oaiglm/__pycache__/
python setup.py install




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






