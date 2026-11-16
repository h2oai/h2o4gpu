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
sudo apt-get purge libatlas-dev libatlas-base-dev
sudo apt install python3-dev
```

If you are building the h2o4gpu R package, it is necessary to install the following dependencies:

```
sudo apt-get -y install libcurl4-openssl-dev libssl-dev libxml2-dev
```

If you want lightgbm support to work, one needs to install boost from source to link statically:

```
# This can be done as a user as well, just change /opt/boost to a user path and avoid sudo.
sudo bash
rm -rf /opt/boost
mkdir -p /opt/boost
export CUDA_HOME=/usr/local/cuda/
apt-get install -y opencl-headers libicu-dev bzip2 libbz2-dev zlib1g-dev python-dev && \
wget https://s3.amazonaws.com/0xdata-public/boost/boost_1_69_0.tar.bz2 && \
tar xjf boost_1_69_0.tar.bz2 && \
cd boost_1_69_0 && \
export PYTHON_PREFIX=`python3-config --prefix` && \
./bootstrap.sh --prefix=/opt/boost/ --with-python=python3 && \
export PYTHON_INCLUDES=`python3-config --includes` && \
export CPPFLAGS="$PYTHON_INCLUDES -fPIC" && \
export C_INCLUDE_PATH="$PYTHON_PREFIX/include/python3.6m/" ; export CPLUS_INCLUDE_PATH="$PYTHON_PREFIX/include/python3.6m/" && \
./b2 link=static -a -d0 install --prefix=/opt/boost/ --with=all -j 20 cxxflags="$CPPFLAGS"
chmod -R a+rx /opt/boost
unset PYTHON_INCLUDES
unset CPPFLAGS
unset CPLUS_INCLUDE_PATH
```

or for dynamic linking remove
```
-DBOOST_ROOT=/opt/boost -DBoost_USE_STATIC_LIBS=1
```
from the Makefile and before compiling do:
```
sudo apt-get -y install ibboost-dev libboost-system-dev libboost-filesystem-dev
```

The default setup is to link statically, as the Dockerfile-build shows, in order to support broader installation types.


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
pip install python-package/dist/xgboost*.whl --upgrade
```
Note: By default XGBoost is built with NCCL support against the system NCCL (`libnccl-devel`).

If fully understand build, can do jump to latter steps of
"fullinstall", but when in doubt always do "fullinstall."

## Re-builds:

"fullinstall" compiles and installs the entire package, which can take a while.  If you only changes certain files, a more limited re-build can be done:

If only change C++ files, can just do:
```
make cpp
```

If only change python files, can just do:
```
make py
```

If only changed how python files were packages, can just do:
```
make install
```

If changed cpp files and want the python install to build and install, do:
```
make build install # same as make cpp py install
```

## Build flags and options:

To find a full list of used flags and options please refer to `make/config.mk`. Here are the most useful ones:

To find a full list of used flags and options please refer to `make/config.mk`. Here are the most useful ones:

##### Debug mode

To build the code in debug mode set `CMAKE_BUILD_TYPE=Debug` when building e.g. `make fullinstall CMAKE_BUILD_TYPE=Debug`.

##### NVTE

To enable `nvToolsExt` set the `USENVTX` variable e.g. `make fullinstall USENVTX=ON`

With `USE_HIP=ON` the same switches emit roctx ranges instead: `USENVTX` links `libroctx64`, and code built with `USE_NVTX` defined (e.g. `-DCMAKE_CXX_FLAGS=-DUSE_NVTX -DCMAKE_HIP_FLAGS=-DUSE_NVTX`) marks its ranges with `roctxRangePushA`/`roctxRangePop`. `rocprofv3 --marker-trace` records them.

##### DEV_BUILD

To expedite the building process in dev setup you can set `DEV_BUILD=ON` e.g. `make fullinstall DEV_BUILD=ON`. This will build the binary with only single CUDA compute capability (currently 6.1).

##### USE_HIP (AMD GPUs)

To build the GPU library for AMD GPUs with ROCm instead of CUDA, install ROCm (with hipBLAS, hipSPARSE, hipSOLVER, hipRAND, hipCUB and rocThrust) and set `USE_HIP=ON`, e.g. `make cpp USE_HIP=ON`. Point CMake at ROCm with `CMAKE_PREFIX_PATH`, or put `/opt/rocm/bin` on `PATH`:

```
export CMAKE_PREFIX_PATH=/opt/rocm
make cpp USE_HIP=ON
make testunit USE_HIP=ON
```

The GPU architecture is detected from the GPUs in the machine. To build for another one, configure directly with CMake:

```
cmake -S . -B build -DUSE_CUDA=OFF -DUSE_HIP=ON -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_HIP_ARCHITECTURES=gfx90a
cmake --build build -j
```

With `USE_HIP=ON`, `make lightgbm_gpu` and `make lightgbm_cpu` build LightGBM from the `LightGBM_rocm` submodule (upstream LightGBM, which supports ROCm with `device_type='cuda'`) with the h2o changes in `patches/LightGBM_rocm` applied, instead of the `LightGBM` submodule.

Likewise `make xgboost` builds XGBoost from the `xgboost_rocm` submodule (AMD's ROCm branch of XGBoost; use `device='cuda'` or `device='gpu'`) with the h2o changes in `patches/xgboost_rocm` applied, instead of the `xgboost` submodule:

```
git submodule update --init --recursive LightGBM_rocm xgboost_rocm
make lightgbm_gpu lightgbm_cpu install_lightgbm_gpu install_lightgbm_cpu USE_HIP=ON
make xgboost install_xgboost USE_HIP=ON
```

The previous XGBoost (`xgboost_prev`) is not built or bundled with `USE_HIP=ON`, so `h2o4gpu.util.xgboost_migration.load_pkl` cannot convert models pickled by XGBoost 0.90 there.

On Windows, configure with CMake 4 and ROCm's `clang-cl` as the C, C++ and HIP compiler, and OpenBLAS from vcpkg. This builds the GPU library, the `_ch2o4gpu_gpu`/`_ch2o4gpu_cpu` Python modules and `h2o4gpu_test.exe`; the Makefile targets and the wheel are Linux-only. Python does not search `PATH` for the modules' DLLs, so add the ROCm `bin` directory with `os.add_dll_directory` before importing them.

```
set ROCM=<ROCm install root>
cmake -G Ninja -S . -B build -DUSE_CUDA=OFF -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DCMAKE_PREFIX_PATH=%ROCM% -DCMAKE_C_COMPILER=%ROCM%/lib/llvm/bin/clang-cl.exe -DCMAKE_CXX_COMPILER=%ROCM%/lib/llvm/bin/clang-cl.exe -DCMAKE_HIP_COMPILER=%ROCM%/lib/llvm/bin/clang-cl.exe -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

##### NCCL

Currently only XGBoost part of the project is using NCCL. By default NCCL support is ON during builds and the system NCCL (`libnccl-devel`, or `NCCL_ROOT` for a non-standard install) is used. To turn it off run with `USENCCL=0` e.g. `make fullinstall USENCCL=0`

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