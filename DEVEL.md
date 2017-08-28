## Developer guide

### Building

#### Build Environment

* Python 3.6.

For `pyenv` go to https://github.com/pyenv/pyenv and follow those instructions for installing pyenv. Then run, e.g.:

````
pyenv install 3.6.1
pyenv global 3.6.1
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

- Install OpenBlas dev environment:

```
sudo apt-get install libopenblas-dev
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
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export VECLIB_MAXIMUM_THREADS=32
```

#### Compiling and Building

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

This installs full h2o4gpu as user. It also compiles a python wheel and puts it in $BASE/src/interface_py/dist/h2o4gpu-0.0.1-py2.py3-none-any.whl .  One can share this wheel and have someone install it as: pip install h2o4gpu-0.0.1-py2.py3-none-any.whl

#### Testing

- test python package
```
make test && make testbig
```

- test performance and accuracy of python package
```
make testperf && make testbigperf
```

- test performance and accuracy of python package for xgboost vs. lightgbm
```
make liblightgbm # only need to do ever once per environment
make testxgboost
```

- show all test errors and timings
```
sh tests/showresults.sh
```

#### Running examples

- Jupyter Notebooks
```
examples/py/demos/H2O4GPU_GLM.ipynb
examples/py/demos/H2O4GPU_GBM.ipynb
examples/py/demos/H2O4GPU_KMeans_Homesite.ipynb
examples/py/demos/H2O4GPU_KMeans_Images.ipynb
```

- To run gpu C++ version:
```
cd $BASE/examples/cpp && make -j all ; make run
```

- Or, to run 16-gpu version on ipums.txt data:
```
./h2o4gpu-glm-gpu-ptr ipums.txt 0 16 16 100 5 5 1 0 0.2 &> fold5x5.txt
```