# Developer guide

## Building

### Build Environment

* Linux machine w/ GCC4.9+ and CMake installed.

* Python 3.6.

For `pyenv` go to https://github.com/pyenv/pyenv and follow those instructions for installing pyenv. Then run, e.g.:

````

env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.6.3
pyenv global 3.6.3
````

For `virtualenv` and ubuntu 16.04:

```arma.header
apt-get -y --no-install-recommends  install \
    python3.6 \
    python3.6-dev \
    virtualenv \
    python3-pip
virtualenv --python=python3.6 .venv
pip install setuptools --no-cache-dir
. .venv/bin/activate
```

- Install required system (apt-get for Ubuntu) dependencies:

```
sudo apt-get install -y libopenblas-dev pbzip2 swig
```

If you are building the h2o4gpu R package, it is necessary to install the following dependencies:

```
sudo apt-get -y install libcurl4-openssl-dev libssl-dev libxml2-dev
```

If you are using `conda`, you probably need to do:
```
conda install libgcc
```

- Add to `.bashrc` or your own environment (e.g.):

```
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH_MORE=/home/$USER/lib/:$CUDA_HOME/lib64/:$CUDA_HOME/lib/:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LD_LIBRARY_PATH_MORE
export CUDADIR=/usr/local/cuda/include/
```

### Compiling and Building

- Do at first the below in order to get GPUs to stay warm to avoid delays upon running h2o4gpu.

```
sudo nvidia-smi -pm 1
```

or

```
sudo nvidia-persistenced --user foo --persistence-mode # where "foo" is your username
```

- To compile everything and install R and python interfaces as user:

```
git clone --recursive git@github.com:h2oai/h2o4gpu.git (or git clone --recursive https://github.com/h2oai/h2o4gpu)
cd h2o4gpu
make fullinstall
```

This installs full h2o4gpu as user. It compiles a python wheel
and puts it in
`$BASE/src/interface_py/dist/h2o4gpu-<h2o4gpu_version>-cp<python_version>-cp<python_version>m-<OS>.whl` .  One
can share this wheel and have someone install it as: `pip install
h2o4gpu-<h2o4gpu_version>-cp<python_version>-cp<python_version>m-<OS>.whl`.

This also downloads and installs the necessary dependencies and then builds the R packge using [install_r_deps.sh](https://github.com/h2oai/h2o4gpu/blob/master/scripts/install_r_deps.sh) script.
You can rebuild the R package using this script.

If already have repository and want to get updates, do:
```
git checkout master
git pull
git submodule update
make fullinstall
```

To compile your own xgboost (instead of using the pip install from aws s3 normally done), do:

```
git clone https://github.com/h2oai/xgboost
cd xgboost
git checkout h2oai
make -f Makefile2
pip install python-package/dist/xgboost-0.71-py3-none-any.whl --upgrade
```
Note: By default the GPU NCCL version is installed using your local cuda version.

If fully understand build, can do jump to latter steps of
"fullinstall", but when in doubt always do "fullinstall."

## Build flags and options:

To find a full list of used flags and options please refer to `make/config.mk`. Here are the most useful ones:

##### Debug mode

To build the code in debug mode set `CMAKE_BUILD_TYPE=Debug` when building e.g. `make fullinstall CMAKE_BUILD_TYPE=Debug`.

##### NVTE

To enable `nvToolsExt` set the `USENVTX` variable e.g. `make fullinstall USENVTX=ON`

##### DEV_BUILD

To expedite the building process in dev setup you can set `DEV_BUILD=ON` e.g. `make fullinstall DEV_BUILD=ON`. This will build the binary with only single CUDA compute capability (currently 6.1).

##### NCCL

Currently only XGBoost part of the project is using NCCL. By default NCCL support is ON during builds. To turn it off run with `USENCCL=0` e.g. `make fullinstall USENCCL=0`

## Testing

- test python package
```
make test
```

- test performance and accuracy of python package
```
make testperf
```

- test performance and accuracy of python package for xgboost vs. lightgbm
```
make liblightgbm # only need to do ever once per environment
make testxgboost
```

- show all test errors and timings (can show only result of testperf or only testxgboost or both):
```
sh tests/showresults.sh
```

## Running examples

- Jupyter Notebooks
```
examples/py/demos/H2O4GPU_GLM.ipynb
examples/py/demos/H2O4GPU_GBM.ipynb
examples/py/demos/H2O4GPU_KMeans_Images.ipynb
```

## Adding code

If you want to contibute code but are not sure where to start have a look at an [example](https://github.com/h2oai/h2o4gpu/tree/master/EXAMPLE_SOLVER.md) showing how to add a new solver from scratch.