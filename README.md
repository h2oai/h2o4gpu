# H2O4GPU

[![Join the chat at https://gitter.im/h2oai/h2o4gpu](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/h2oai/h2o4gpu)

**H2O4GPU** is a collection of GPU solvers by [H2Oai](https://www.h2o.ai/) with APIs in Python and R.  The Python API builds upon the easy-to-use [scikit-learn](http://scikit-learn.org) API and its well-tested CPU-based algorithms.  It can be used as a drop-in replacement for scikit-learn (i.e. `import h2o4gpu as sklearn`) with support for GPUs on selected (and ever-growing) algorithms.  H2O4GPU inherits all the existing scikit-learn algorithms and falls back to CPU algorithms when the GPU algorithm does not support an important existing scikit-learn class option.  The R package is a wrapper around the H2O4GPU Python package, and the interface follows standard R conventions for modeling.


Daal library added for CPU, currently supported only x86_64 architecture.

## Requirements

* PC running Linux with glibc 2.17+

* Install CUDA with bundled display drivers (
  [CUDA 8](https://docs.nvidia.com/cuda/archive/8.0/cuda-installation-guide-linux/index.html)
  or
  [CUDA 9](https://docs.nvidia.com/cuda/archive/9.0/cuda-installation-guide-linux/index.html)
  or
  [CUDA 9.2](https://docs.nvidia.com/cuda/archive/9.2/cuda-installation-guide-linux/index.html))

* Python shared libraries (e.g. On Ubuntu:  sudo apt-get install libpython3.6-dev)

When installing, choose to link the cuda install to /usr/local/cuda .
Ensure to reboot after installing the new nvidia drivers.

* Nvidia GPU with Compute Capability >= 3.5 ([Capability Lookup](https://developer.nvidia.com/cuda-gpus)).

* For advanced features, like handling rows/32 > 2^16 (i.e., rows > 2,097,152) in K-means, need Capability >= 5.2

* For building the R package, `libcurl4-openssl-dev`, `libssl-dev`, and `libxml2-dev` are needed.

## User Installation

Note: Installation steps mentioned below are for users planning to use H2O4GPU. See [DEVEL.md](DEVEL.md) for developer installation.

H2O4GPU can be installed using either PIP or Conda


### Prerequisites
Add to `~/.bashrc` or environment (set appropriate paths for your OS):

```
export CUDA_HOME=/usr/local/cuda # or choose /usr/local/cuda9 for cuda9 and /usr/local/cuda8 for cuda8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64/:$CUDA_HOME/lib/:$CUDA_HOME/extras/CUPTI/lib64
```

- Install OpenBlas dev environment:

```
sudo apt-get install libopenblas-dev pbzip2
```

If you are building the h2o4gpu R package, it is necessary to install the following dependencies:

```
sudo apt-get -y install libcurl4-openssl-dev libssl-dev libxml2-dev
```

### PIP install 
Download the Python wheel file (For Python 3.6):

  * Stable:
    * [CUDA8 - linux_x86_64](https://s3.amazonaws.com/h2o-release/h2o4gpu/releases/stable/ai/h2o/h2o4gpu/0.3-cuda8/h2o4gpu-0.3.1.10000-cp36-cp36m-linux_x86_64.whl)
    * [CUDA9.0 - linux_x86_64](https://s3.amazonaws.com/h2o-release/h2o4gpu/releases/stable/ai/h2o/h2o4gpu/0.3-cuda9/h2o4gpu-0.3.1.10000-cp36-cp36m-linux_x86_64.whl)
    * [CUDA9.2 - linux_x86_64](https://s3.amazonaws.com/h2o-release/h2o4gpu/releases/stable/ai/h2o/h2o4gpu/0.3-cuda92/h2o4gpu-0.3.1.10000-cp36-cp36m-linux_x86_64.whl)
     * [CUDA8 - linux_ppc64le](https://s3.amazonaws.com/h2o-release/h2o4gpu/releases/stable/ai/h2o/h2o4gpu/0.3-cuda8/h2o4gpu-0.3.1.10000-cp36-cp36m-linux_ppc64le.whl)
    * [CUDA9.0 - linux_ppc64le](https://s3.amazonaws.com/h2o-release/h2o4gpu/releases/stable/ai/h2o/h2o4gpu/0.3-cuda9/h2o4gpu-0.3.1.10000-cp36-cp36m-linux_ppc64le.whl)
    * [CUDA9.2 - linux_ppc64le](https://s3.amazonaws.com/h2o-release/h2o4gpu/releases/stable/ai/h2o/h2o4gpu/0.3-cuda92/h2o4gpu-0.3.1.10000-cp36-cp36m-linux_ppc64le.whl)
  * Bleeding edge (changes with every successful master branch build):
    * [CUDA8 - linux_x86_64](https://s3.amazonaws.com/h2o-release/h2o4gpu/releases/bleeding-edge/ai/h2o/h2o4gpu/0.3-cuda8/h2o4gpu-0.3.1.10000-cp36-cp36m-linux_x86_64.whl)
    * [CUDA9.0 - linux_x86_64](https://s3.amazonaws.com/h2o-release/h2o4gpu/releases/bleeding-edge/ai/h2o/h2o4gpu/0.3-cuda9/h2o4gpu-0.3.1.10000-cp36-cp36m-linux_x86_64.whl)
    * [CUDA9.2 - linux_x86_64](https://s3.amazonaws.com/h2o-release/h2o4gpu/releases/bleeding-edge/ai/h2o/h2o4gpu/0.3-cuda92/h2o4gpu-0.3.1.10000-cp36-cp36m-linux_x86_64.whl)
    * [CUDA8 - linux_ppc64le](https://s3.amazonaws.com/h2o-release/h2o4gpu/releases/bleeding-edge/ai/h2o/h2o4gpu/0.3-cuda8/h2o4gpu-0.3.1.10000-cp36-cp36m-linux_ppc64le.whl)
    * [CUDA9.0 - linux_ppc64le](https://s3.amazonaws.com/h2o-release/h2o4gpu/releases/bleeding-edge/ai/h2o/h2o4gpu/0.3-cuda9/h2o4gpu-0.3.1.10000-cp36-cp36m-linux_ppc64le.whl)
    * [CUDA9.2 - linux_ppc64le](https://s3.amazonaws.com/h2o-release/h2o4gpu/releases/bleeding-edge/ai/h2o/h2o4gpu/0.3-cuda92/h2o4gpu-0.3.1.10000-cp36-cp36m-linux_ppc64le.whl)

 Start a fresh pyenv or virtualenv session.

Install the Python wheel file. NOTE: If you don't use a fresh environment, this will
overwrite your py3nvml and xgboost installations to use our validated
versions.

```
pip install h2o4gpu-0.3.0-cp36-cp36m-linux_x86_64.whl
```

### Conda installation

Ensure you meet the Requirements and have installed the Prerequisites.

If not already done you need to [install conda package manager](https://conda.io/docs/user-guide/install/linux.html). Ensure you [test your conda installation](https://conda.io/docs/user-guide/install/test-installation.html)

H204GPU packages for CUDA8, CUDA 9 and CUDA 9.2 are available from [h2oai channel in anaconda cloud](https://anaconda.org/h2oai). 

Create a new conda environment with H2O4GPU based on CUDA 9.2 and all its dependencies using the following command. For other cuda versions substitute the package name as needed. Note the requirement for h2oai and conda-forge channels. 

```bash
conda create -n h2o4gpuenv -c h2oai -c conda-forge h2o4gpu-cuda92
```

Once the environment is created activate it `source activate h2o4gpuenv`. 

To test, start an interactive python session in the environment and follow the steps in the Test Installation section below.

### h2o4gpu R package

At this point, you should have installed the H2O4GPU Python package successfully. You can then go ahead and install the `h2o4gpu` R package via the following:

```r
if (!require(devtools)) install.packages("devtools")
devtools::install_github("h2oai/h2o4gpu", subdir = "src/interface_r")
```

Detailed instructions can be found [here](https://github.com/h2oai/h2o4gpu/tree/master/src/interface_r).



## Test Installation

To test your installation of the Python package, the following code:

```
import h2o4gpu
import numpy as np

X = np.array([[1.,1.], [1.,4.], [1.,0.]])
model = h2o4gpu.KMeans(n_clusters=2,random_state=1234).fit(X)
model.cluster_centers_
```
should give input/output of:
```
>>> import h2o4gpu
>>> import numpy as np
>>>
>>> X = np.array([[1.,1.], [1.,4.], [1.,0.]])
>>> model = h2o4gpu.KMeans(n_clusters=2,random_state=1234).fit(X)
>>> model.cluster_centers_
array([[ 1.,  1.  ],
       [ 1.,  4.  ]])
```

To test your installation of the R package, try the following example that builds a simple [XGBoost](https://github.com/dmlc/xgboost) random forest classifier:

``` r
library(h2o4gpu)

# Setup dataset
x <- iris[1:4]
y <- as.integer(iris$Species) - 1

# Initialize and train the classifier
model <- h2o4gpu.random_forest_classifier() %>% fit(x, y)

# Make predictions
predictions <- model %>% predict(x)
```

## Next Steps

For more examples using Python API, please check out our [Jupyter notebook demos](https://github.com/h2oai/h2o4gpu/tree/master/examples/py/demos). To run the demos using a local wheel run, at least download `src/interface_py/requirements_runtime_demos.txt` from the Github repo and do:
```
pip install -r src/interface_py/requirements_runtime_demos.txt
```
and then run the jupyter notebook demos.

For more examples using R API, please visit the [vignettes](https://github.com/h2oai/h2o4gpu/tree/master/src/interface_r/vignettes).

## Running Jupyter Notebooks

You can run Jupyter Notebooks with H2O4GPU in the below two ways

### Creating a Conda Environment

Ensure you have a machine that meets the Requirements and Prerequisites mentioned above. 

Next follow Conda installation instructions mentioned above. Once you have activated the environment, you will need to downgrade tornado to version 4.5.3 [refer issue #680](https://github.com/h2oai/h2o4gpu/issues/680). Start Jupyter notebook, and navigate to the URL shown in the log output in your browser. 

```bash
source activate h2o4gpuenv
conda install tornado==4.5.3
jupyter notebook --ip='*' --no-browser
```
Start a Python 3 kernel, and try the code in [example notebooks](https://github.com/h2oai/h2o4gpu/tree/master/examples/py/demos)

### Using precompiled docker image

Requirements:

* Nvidia drivers compatible with CUDA version used (e.g. 384+ for CUDA9)
* [docker-ce 17](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/)
* [nvidia-docker 1.0](https://github.com/NVIDIA/nvidia-docker/tree/1.0)

Download the Docker file (for linux_x86_64):

  * Bleeding edge (changes with every successful master branch build):
    * [CUDA8](https://s3.amazonaws.com/h2o-release/h2o4gpu/releases/bleeding-edge/ai/h2o/h2o4gpu/0.3-cuda8/h2o4gpu-0.3.0.10000-cuda8-runtime.tar.bz2)
    * [CUDA9](https://s3.amazonaws.com/h2o-release/h2o4gpu/releases/bleeding-edge/ai/h2o/h2o4gpu/0.3-cuda9/h2o4gpu-0.3.0.10000-cuda9-runtime.tar.bz2)
    * [CUDA9.2](https://s3.amazonaws.com/h2o-release/h2o4gpu/releases/bleeding-edge/ai/h2o/h2o4gpu/0.3-cuda92/h2o4gpu-0.3.0.10000-cuda92-runtime.tar.bz2)

Load and run docker file (e.g. for bleeding-edge of cuda92):

By default, remote access is disabled in the docker images. If remote access to Jupyter notebooks is needed follow the below instructions to create a config file locally and mount it to the docker container to override the default settings. 
    
```
jupyter notebook --generate-config
echo "c.NotebookApp.allow_remote_access = True >> ~/.jupyter/jupyter_notebook_config.py
pbzip2 -dc h2o4gpu-0.3.0.10000-cuda92-runtime.tar.bz2 | nvidia-docker load
mkdir -p log
nvidia-docker run --name localhost --rm -p 8888:8888 -u `id -u`:`id -g` -v `pwd`/log:/log -v /home/$USER/.jupyter:/jupyter --entrypoint=./run.sh opsh2oai/h2o4gpu-0.3.0.10000-cuda92-runtime &
find log -name jupyter* -type f -printf '%T@ %p\n' | sort -k1 -n | awk '{print $2}' | tail -1 | xargs cat | grep token | grep http | grep -v NotebookApp
```

If you do not need remote access, follow the below instructions to start the container with default settings

```
pbzip2 -dc h2o4gpu-0.3.0.10000-cuda92-runtime.tar.bz2 | nvidia-docker load
mkdir -p log
nvidia-docker run --name localhost --rm -p 8888:8888 -u `id -u`:`id -g` -v `pwd`/log:/log --entrypoint=./run.sh opsh2oai/h2o4gpu-0.3.0.10000-cuda92-runtime &
find log -name jupyter* -type f -printf '%T@ %p\n' | sort -k1 -n | awk '{print $2}' | tail -1 | xargs cat | grep token | grep http | grep -v NotebookApp
```

Copy/paste the http link shown into your browser.  If the "find" command doesn't work, look for the latest jupyter.log file and look at contents for the http link and token.

If the link shows no token or shows ... for token, try a token of "h2o" (without quotes).  If running on your own host, the weblink will look like http://localhost:8888:token with token replaced by the actual token.

This container has a /demos directory which contains Jupyter notebooks and some data.

## Plans

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

Another primary goal is to support all operations on the GPU via the
[GOAI
initiative](https://devblogs.nvidia.com/parallelforall/goai-open-gpu-accelerated-data-analytics/).
This involves ensuring the GPU algorithms can take and return GPU
pointers to data instead of going back to the host.  In scikit-learn
API language these are called fit\_ptr, predict\_ptr, transform\_ptr,
etc., where ptr stands for memory pointer.


## RoadMap
### 2019 Q2:
* A new processing engine that allows to scale beyond GPU memory limits
* k-Nearest Neighbors
* Matrix Factorization
* Factorization Machines
* API Support: GOAI API support
* Data.table support

More precise information can be found in the [milestone's list](https://github.com/h2oai/h2o4gpu/milestones).

## Solver Classes

Among others, the solver can be used for the following classes of problems

  + GLM: Lasso, Ridge Regression, Logistic Regression, Elastic Net Regulariation
  + KMeans
  + Gradient Boosting Machine (GBM) via [XGBoost](https://devblogs.nvidia.com/parallelforall/gradient-boosting-decision-trees-xgboost-cuda/)
  + Singular Value Decomposition(SVD) + Truncated Singular Value Decomposition
  + Principal Components Analysis(PCA)

## Benchmarks

Our benchmarking plan is to clearly highlight when modeling benefits
from the GPU (usually complex models) or does not (e.g. one-shot
simple models dominated by data transfer).

We have benchmarked h2o4gpu, scikit-learn, and h2o-3 on a variety of
solvers.  Some benchmarks have been performed for a few selected cases
that highlight the GPU capabilities (i.e. compute or on-GPU memory
operations dominate data transfer to GPU from host):

[Benchmarks for GLM, KMeans, and XGBoost for CPU vs. GPU.](https://github.com/h2oai/h2o4gpu/blob/master/presentations/benchmarks.pdf)

A suite of benchmarks are computed when doing "make testperf" from a
build directory. These take all of our tests and benchmarks h2o4gpu
against h2o-3.  These will soon be presented as a live
commit-by-commit streaming plots on a website.


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
