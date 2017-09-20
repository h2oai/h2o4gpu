# H2O4GPU

[![Join the chat at https://gitter.im/h2oai/h2o4gpu](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/h2oai/h2o4gpu)

## FAQ

### Should I expect identical answers to sklearn? ###

One should expect answers similar to within the requested tolerance.

For example, for GLM, a pure L2 problem will converge to similar
answers to within the tolerance, but L1 could select different columns
that happens to be similar to within the tolerance.

For example, for KMeans, if one feeds the cluster positions that are
computed by the GPU algorithm and feed them into sklearn, the
resulting cluster positions shouldn't change except within tolerance.

For example, for xgboost, the method used to split is slightly
different, so one shouldn't expect identical results to sklearn.

### What are the benchmark results?  Are there public test results? ###

The developer environment has "make test" and "make testperf", the
latter which provides table data of timing and accuracy comparisons
with h2o-3.

In general, if the data is small enough, the CPU alorithm can be
faster than the GPU algorithm, because the time to transfer data
to/from the GPU dominates the time.  For large enough data sizes, the
GPU is often faster.  Eventually, the GPU memory can be exhausted, at
which point the GPU algorithm may simply fail with a cuda system error
or memory error.

### Are the results reproducible? Do we have tests for this? ###

We are developing reproducibility tests.

### Can we run algos on more than one gpu? ###

See the current roadmap: [README](https://github.com/h2oai/h2o4gpu/tree/master/README.md)

### Is it distributed across multiple nodes? ###

Not yet.

### What happens if data does not fit into GPU memory? Is there a CPU fall back? ###

Currently there is no automatic fallback to the CPU algorithm in the
event of memory filling up.

### How do we handle missing values? Categoricals? Is one hot encoding needed for categoricals? Missing levels for categoricals?How do we handle missing values? Categoricals? Is one hot encoding needed for categoricals? Missing levels for categoricals? ###

Array inputs must be pandas or numpy and completely numerical.  No
missings or categoricals are currently allowed.  So all munging must
be done using other algorithms, such as those included in sklearn
itself (that h2o4gpu inherits).

### How to install python virtual environment ###

For pyenv, you can follow instructions at: [pyenv](https://github.com/pyenv/pyenv/).

Instead of default way to install, do instead the below to get the shared python libraries needed by numba:

PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.6.1

### How can I force use of the GPU back end no matter how I set my class parameters ###

There are two options:

* Add a named argument of backend='h2o4gpu' to each class wrapper
  (those in h2o4gpu.solvers like h2o4gpu.solvers.LogisticRegression).

* Before calling python or jupyter notebook, set the environment variable
H2O4GPU_BACKEND to 'h2o4gpu' .

To force sklearn always, set these to 'sklearn'.

The default is backend='auto', and we try to make reasonable decisions
about which back end is used.

### How is this different from scikit-cuda,  pycuda, magma, cula, etc. ###

[Scikit-cuda](https://github.com/lebedov/scikit-cuda) and
[pycuda](https://pypi.python.org/pypi/pycuda) are both python wrappers
for underlying cuda C++ functions.  They make it easy for python users
to obtain the functionality of C++ cuda functions.  The cuda functions
perform many linear algebra operations (CUBLAS, CUSOLVER) and other
specific algorithms (CUFFT).

[MAGMA](http://icl.cs.utk.edu/magma/) is a set of linear algebra and
other mathematical routines, and includes some basic fitting routines
(without regularization), and it has good support for using either
CPUs, GPUs, or multiple GPUs in some cases.

[CULA](http://www.culatools.com/) is a closed-source library that
provides matrix operations and linear algebra operations.
[Arrayfire](https://github.com/arrayfire/arrayfire) has a variety of
image processing, statistical, and linear algebra algorithms with a
diverse support of different back-end hardware.

[H2O4PU](https://github.com/h2oai/h2o4gpu) focuses on machine learning
(ML) algorithms that can be used in artificial intelligence (AI)
contexts.  ML algorithms differ from standard linear algebra
algorithms, because ML requires unsupervised and supervised modeling
with regularization and validation.

It has a C++ back-end that uses many different cuda, cub,
and thrust functions. Since H2O4PU already uses C++ as a backend, the
scikit-cuda, pycuda, and CULA packages don't provide any additional
functionality (although in limited use, they may make it easier to
prototype new features).

In addition, by having H2O4GPU do all operations with C++, this allows
the data to stay on the GPU for the entire ML algorithm, instead of
data being transferred back and forth between the CPU and GPU (an
expensive operation) for each separate linear algebra call.  This
allows the ML algorithm to be designed to have all inputs and outputs
on the GPU, allow the ML algorithm to be a component within a pure GPU
pipeline.


