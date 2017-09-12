# H2O4GPU

[![Join the chat at https://gitter.im/h2oai/h2o4gpu](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/h2oai/h2o4gpu)

## FAQ

### Should I expect identical answers to sklearn? ###

One should expect answerss similar to within the requested tolerance.

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

### How do we handle missing values? Categoricals? Is one hot encoding
  needed for categoricals? Missing levels for categoricals?How do we
  handle missing values? Categoricals? Is one hot encoding needed for
  categoricals? Missing levels for categoricals? ###

The input data must be pandas or numpy and completely numerical.  No
missings or categoricals are currently allowed.  So all munging must
be done using other algorithms, such as those included in sklearn
itself (that h2o4gpu inherits).

