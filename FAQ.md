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

### How is this different from scikit-cuda or pycuda ###

[Scikit-cuda](https://github.com/lebedov/scikit-cuda) and
[pycuda](https://pypi.python.org/pypi/pycuda) are both python wrappers
for underlying cuda C++ functions.  They make it easy for python users
to obtain the functionality of C++ cuda functions.  The cuda functions
perform many linear algebra operations (CUBLAS, CUSOLVER) and other
specific algorithms (CUFFT).

H2O4PU has a C++ back-end that uses many different cuda, cub, and
thrust functions to provide machine learning (ML) algorithms that can
be used in artificial intelligence (AI) applications.  Since H2O4PU
already uses C++ as a backend, these packages don't provide any
additional functionality.

In addition, by having H2O4GPU do all operations with C++, this allows
the data to stay on the GPU for the entire ML algorithm, instead of
data being transferred back and forth between the CPU and GPU (an
expensive operation) for each separate linear algebra call.

Lastly, this allows the ML algorithm to be designed to have all inputs
and outputs on the GPU, allow the ML algorithm to be a component
within a pure GPU pipeline.
