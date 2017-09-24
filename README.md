# H2O4GPU

[![Join the chat at https://gitter.im/h2oai/h2o4gpu](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/h2oai/h2o4gpu)

**H2O4GPU** is a collection of GPU solvers by H2Oai.  It builds upon
the easy-to-use [scikit-learn](modules/ensemble.html#forest) API and its well-tested CPU-based
algorithms.  It can be used as a drop-in replacement for scikit-learn
(i.e. `import h2o4gpu as sklearn`) with support for GPUs on selected
(and ever-growing) algorithms.  H2OG4PU inherits all the existing
scikit-learn algorithms and falls-back to CPU aglorithms when the GPU
algorithm does not support an important existing Scikit-learn class
option.

An R API is in developement and will be released as a stand-alone R package in the future.

## Requirements

* PC with Ubuntu 16.04+

* Install CUDA with bundled display drivers
  [CUDA 8](https://developer.nvidia.com/cuda-downloads)
  or
  [CUDA 9](https://developer.nvidia.com/cuda-release-candidate-download)

When installing, choose to link the cuda install to /usr/local/cuda .
Ensure to reboot after installing the new nvidia drivers.

* Nvidia GPU with Compute Capability>=3.5 [Capability Lookup](https://developer.nvidia.com/cuda-gpus).

## Installation

Add to `~/.bashrc` or environment (set appropriate paths for your OS):

```
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64/:$CUDA_HOME/lib/:$CUDA_HOME/extras/CUPTI/lib64
```

- Install OpenBlas dev environment:

```
sudo apt-get install libopenblas-dev
```

Download the Python wheel file:

  * [Stable: Python 3.6, CUDA 8](https://s3.amazonaws.com/artifacts.h2o.ai/releases/stable/ai/h2o/h2o4gpu/0.0.4/h2o4gpu-0.0.4-py36-none-any.whl)
  * [Bleeding edge: Python 3.6, CUDA 8](https://s3.amazonaws.com/artifacts.h2o.ai/releases/bleeding-edge/ai/h2o/h2o4gpu/0.0.4/h2o4gpu-0.0.4-py36-none-any.whl)
  * [Bleeding edge: Conda Python 3.6, CUDA 8](https://s3.amazonaws.com/artifacts.h2o.ai/releases/bleeding-edge/ai/h2o/h2o4gpu/0.0.4_conda/h2o4gpu-0.0.4-py36-none-any.whl)
 
Start a fresh pyenv or virtualenv session.

Install the Python wheel file. NOTE: If you don't use a fresh environment, this will
overwrite your py3nvml and xgboost installations to use our validated
versions.

```
pip install h2o4gpu-0.0.4-py36-none-any.whl
```

Test your installation

```
import h2o4gpu
import numpy as np

X = np.array([[1.,1.], [1.,4.], [1.,0.]])
model = h2o4gpu.KMeans(n_clusters=2,random_state=1234).fit(X)
model.fit(X).cluster_centers_
```
Should give input/output of:
```
>>> import h2o4gpu
>>> import numpy as np
>>>
>>> X = np.array([[1.,1.], [1.,4.], [1.,0.]])
>>> model = h2o4gpu.KMeans(n_clusters=2,random_state=1234).fit(X)
Copying centroid data to device: 1
Copying centroid data to device: 2
>>> model.fit(X).cluster_centers_
Copying centroid data to device: 1
Copying centroid data to device: 2
array([[ 0.25,  0.  ],
       [ 1.  ,  4.  ]])
```

For more examples check our [Jupyter notebook demos](https://github.com/h2oai/h2o4gpu/tree/master/examples/py/demos).

## Plans and RoadMap

The vision is to develop fast GPU algorithms to complement the CPU
algorithms in scikit-learn while keeping full scikit-learn API
compatibility and scikit-learn CPU algorithm capability. The h2o4gpu
Python module is to be used as a drop-in-replacement for scikit-learn
that has the full functionality of scikit-learn's CPU algorithms.

Functions and classes will be gradually overridden by GPU-enabled algorithms (unless
`n_gpu=0` is set and we have no CPU algorithm except scikit-learn's).
The CPU algorithms and code initially will be sklearn, but gradually
those may be replaced by faster open-source codes like those in Intel
DAAL.

This vision is currently accomplished by using the open-source
scikit-learn and xgboost and overriding scikit-learn calls with our
own GPU versions.  In cases when our GPU class is currently
incapable of an important scikit-learn feature, we revert to the
scikit-learn class.

As noted above, there is an R API in development, which will be
released as a stand-alone R package.  All algorithms supported by
H2O4GPU will be exposed in both Python and R in the future.

Another primary goal is to support all operations the GPU via the
[GOAI
initiative](https://devblogs.nvidia.com/parallelforall/goai-open-gpu-accelerated-data-analytics/).
This involves ensuring the GPU algorithms can take and return GPU
pointers to data instead of going back to the host.  In scikit-learn
API language these are called fit\_ptr, predict\_ptr, transform\_ptr,
etc., where ptr stands for memory pointer.


![Alt text](https://github.com/h2oai/h2o4gpu/blob/master/roadmap.jpg
 "ROADMAP.")

## Solver Classes

Among others, the solver can be used for the following classes of problems

  + GLM: Lasso, Ridge Regression, Logistic Regression, Elastic Net Regulariation,
  + KMeans
  + Gradient Boosting Machine (GBM) via [XGBoost](https://devblogs.nvidia.com/parallelforall/gradient-boosting-decision-trees-xgboost-cuda/)

Planned:
  + GLM: Linear SVM, Huber Fitting, Total Variation Denoising, Optimal Control, Linear Programs and Quadratic Programs.
  + SVD, PCA

## Benchmarks

[Benchmarks for GLM, KMeans, and XGBoost for CPU vs. GPU.](https://github.com/h2oai/h2o4gpu/blob/master/presentations/benchmarks.pdf)
 

## Contributing

Please refer to our [CONTRIBUTING.md](CONTRIBUTING.md) and
[DEVEL.md](DEVEL.md) for instructions on how to build and test the
project and how to contribute.  The h2o4gpu
[Gitter](https://gitter.im/h2oai/h2o4gpu) chatroom can be used for
discussion related to open source development.

GitHub [issues](https://github.com/h2oai/h2o4gpu/issues) are used for bugs, feature and enhancement discussion/tracking.



## Questions

* Please ask all code-related questions on [StackOverflow](https://stackoverflow.com/questions/tagged/h2o4gpu) using the "h2o4gpu" tag.  

* Questions related to the roadmap can be directed to the developers on [Gitter](https://gitter.im/h2oai/h2o4gpu).

* [Troubleshooting](https://github.com/h2oai/h2o4gpu/tree/master/TROUBLESHOOTING.md)

* [FAQ](https://github.com/h2oai/h2o4gpu/tree/master/FAQ.md)


## References

1. [Parameter Selection and Pre-Conditioning for a Graph Form Solver -- C. Fougner and S. Boyd][pogs]
2. [Block Splitting for Distributed Optimization -- N. Parikh and S. Boyd][block_splitting]
3. [Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers -- S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein][admm_distr_stats]
4. [Proximal Algorithms -- N. Parikh and S. Boyd][prox_algs]


[pogs]: http://stanford.edu/~boyd/papers/pogs.html "Parameter Selection and Pre-Conditioning for a Graph Form Solver -- C. Fougner and S. Boyd"

[block_splitting]: http://www.stanford.edu/~boyd/papers/block_splitting.html "Block Splitting for Distributed Optimization -- N. Parikh and S. Boyd"

[admm_distr_stats]: http://www.stanford.edu/~boyd/papers/block_splitting.html "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers -- S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein"

[prox_algs]: http://www.stanford.edu/~boyd/papers/prox_algs.html "Proximal Algorithms -- N. Parikh and S. Boyd"

## Copyright

```
Copyright (c) 2017, H2O.ai, Inc., Mountain View, CA
Apache License Version 2.0 (see LICENSE file)


This software is based on original work under BSD-3 license by:

Copyright (c) 2015, Christopher Fougner, Stephen Boyd, Stanford University
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
