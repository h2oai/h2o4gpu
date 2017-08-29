# H2O4GPU

[![Join the chat at https://gitter.im/h2oai/h2o4gpu](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/h2oai/h2o4gpu)

**H2O4GPU** is a collection of GPU (and CPU) solvers by H2Oai, as drop-in replacement of sklearn with GPU capabilities.

## Requirements

* Install [CUDA 8](https://developer.nvidia.com/cuda-downloads).

## Installation

Add to `~/.bashrc` or environment (set appropriate paths for your OS):

```
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64/:$CUDA_HOME/lib/:$CUDA_HOME/extras/CUPTI/lib64
```

Download the Python wheel file
  * (TBA) [Stable](https://s3.amazonaws.com/artifacts.h2o.ai/releases/stable/ai/h2o/h2o4gpu/0.0.3/h2o4gpu-0.0.3-py2.py3-none-any.whl)
  * [Bleeding edge](https://s3.amazonaws.com/artifacts.h2o.ai/releases/bleeding-edge/ai/h2o/h2o4gpu/0.0.3/h2o4gpu-0.0.3-py2.py3-none-any.whl)
 
Install the Python wheel file:

```
pip install h2o4gpu-0.0.3-py2.py3-none-any.whl
```

Test your installation

```
import h2o4gpu
import numpy as np

X = np.array([[1.,1.], [1.,4.], [1.,0.]])
model = h2o4gpu.KMeans(n_clusters=2).fit(X)
model.fit(X).cluster_centers_
```

For more examples check our [Jupyter notebook demos](https://github.com/h2oai/h2o4gpu/tree/master/examples/py/demos).

## Plans and RoadMap

Vision is to have a drop-in replacement for scikit-learn that has the full functionality of sklearn, but gradually modules or classes are replaced by GPU-enabled algorithms.

![Alt text](https://github.com/h2oai/h2o4gpu/blob/master/roadmap.jpg "ROADMAP.")

## Solver Classes

Among others, the solver can be used for the following classes of problems

  + GLM: Lasso, Ridge Regression, Logistic Regression, Elastic Net Regulariation,
  + KMeans

Planned:
  + GLM: Huber Fitting, Total Variation Denoising, Optimal Control, Linear Programs and Quadratic Programs.
  + SVD, PCA


## Contributing

Please refer to our [CONTRIBUTING.md](CONTRIBUTING.md) and [DEVEL.md](DEVEL.md) for instructions on how to build and test the project and how to contribute.

GitHub issues are used only for bugs, feature and enhancement discussion/tracking.

## Questions

Please ask all `h2o4gpu` related questions either on [StackOverflow](https://stackoverflow.com/questions/tagged/h2o4gpu) or our [Gitter](https://gitter.im/h2oai/h2o4gpu),

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