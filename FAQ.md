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

### Why is the GPU algorithm so slow? ###

Ensure the model.model.backend is 'h2o4gpu' to confirm the GPU was
used.  You should also get a WARNING that the class reverted to
sklearn.

Check your input data type by checking model.model.double_precision
(or directly model.double_precision if not using wrapper), and if this
is 1 uses 64-bit floats on the GPU.  64-bit floats are much slower
than 32-bit floats (double_precision=0).

For small data sizes, the transfer time to the GPU can take
longer than the processing on the GPU, and the CPU can be faster in
such or similar cases.

So, the GPU is most useful when doing multiple models on the same
data, like full alpha-lambda path in elastic net, large k in k-means,
or multiple trees in GBM or RF.

Some sklearn algorithms, like their Ridge solver, uses specialized
algorithms that can solve that particular case quickly (because it's a
simple smooth loss function or has an exact closed-form solution).
These specialized algorithms can be ported to the GPU to obtain much
faster solver times.  Even fast algorithms, like our GPU version of
elastic net, can be improved to use parallel coordinate descent for an
even faster GPU algorithm.

### Why is sklearn bundled into h2o4gpu? ###

Minor reason 1) Our wrappers need API consistency, and sometimes
sklearn or xgboost change their API.  Or we want to support advanced
featuers in the APIs that are by default disabled.

Minor reason 2) We want to ensure anyone who uses h2o4gpu gets a
consistent and reliable experience in terms of stability.  Anyone can
rebuild h2o4gpu against any sklearn or xgboost version they like, but
it wouldn't be validated to work.

Major reason: We found it easiest to wrap our GPU backend and the
sklearn as a backend by having a wrapper class that is the same name
as sklearn classes.  And in order to easily inherit sklearn CPU
backends, even if we haven't written a wrapper, bundling seems easiest
via a bit of bash magic.

### Will we create wrappers for all sklearn functions ###

We will rely upon the community developers to create any additional
wrappers.  The scikit-learn design decision to have many wrappers that
call very similar backend classes leads to excessive code that becomes
difficult to maintain.  We have created wrappers for several cases,
but we will focus development on improving the primary classes and
adding advanced features rather than creating individual wrappers that
only vary by one or a few parameters of the back end class.

In cases where some base classes completely reproduce scikit-learn, we
can remove the base wrapper and any sub-classes in scikit-learn will
automatically use the base class.

### How is this different from scikit-cuda,  pycuda, magma, cula, etc. ###

[Scikit-cuda](https://github.com/lebedov/scikit-cuda) and
[pycuda](https://pypi.python.org/pypi/pycuda) are both python wrappers
for underlying cuda C++ functions.  They make it easy for python users
to obtain the functionality of C++ cuda functions.  The cuda functions
perform many linear algebra operations (CUBLAS, CUSOLVER) and other
specific algorithms (CUFFT).

[Numba](https://numba.pydata.org/) is like python numpy for the GPU
and allows compiling python functions for the GPU.  One can use numba
with pyculib, which allows access to native C++ cuda functions,
similar to scikit-cuda.

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

### How is this different from GPUMLib, etc. ###

[GPUMLib](http://gpumlib.sourceforge.net/) has ANN, SVM and matrix factorization algorithms,
[GTSVM](http://ttic.uchicago.edu/~cotter/projects/gtsvm/) has kernelized SVMs, and
[cuSVM](http://patternsonascreen.net/cuSVM.html) has SVMs for classification and regression.


### How can I use pygdf with h2o4gpu?

### How can I use pygdf with h2o4gpu inside DAI environment?

1) Get DAI cuda9.0 rpm/deb installed
```
wget https://s3.amazonaws.com/artifacts.h2o.ai/releases/ai/h2o/dai/rel-1.2.2-6/x86_64-centos7/dai_1.2.2_amd64.deb
dpkg -i dai_1.2.2_amd64.deb
apt-get update
apt-get install dai_1.2.2_amd64.deb
```

2) Install gdf libs:

```
mkdir gdf ; cd gdf
# Go to:
https://anaconda.org/gpuopenanalytics/libgdf/files
# Download https://anaconda.org/gpuopenanalytics/libgdf/0.1.0a2.dev/download/linux-64/libgdf-0.1.0a2.dev-cuda9.0_67.tar.bz2
tar jxvf libgdf-0.1.0a2.dev-cuda9.0_67.tar.bz2
# This will create three folders in the current directory, lib, include and info. Ignore the info folder. We need only lib and include for the next step
sudo cp -a include/gdf /opt/h2oai/dai/python/include/
sudo cp -a lib/libgdf.so /opt/h2oai/dai/python/lib
sudo chmod a+rx /opt/h2oai/dai/python/lib/libgdf.so
```

3) Make and install wheel files

```
# Make wheel files
git clone https://github.com/gpuopenanalytics/pygdf.git
# Created a docker container for building the product
cd pygdf/ ; docker build -t pygdf .
# Logged in interactively to the image:
mkdir -p ~/tmpw ; chmod u+rwx ~/tmpw ; docker run -it -v ~/tmpw:/tmpw pygdf bash
# Activate the conda environment
cd /
source activate gdf
cd pygdf
python setup.py bdist_wheel
cp dist/*.whl /tmpw/
cd /libgdf/build
cmake ..
make install
make copy_python
python setup.py bdist_wheel
cp dist/*.whl /tmpw/
exit

# Now install the wheel files
cd ~/tmpw/
sudo /opt/h2oai/dai/dai-env.sh python -m wheel install `ls libgdf_cffi*.whl` --force
sudo /opt/h2oai/dai/dai-env.sh python -m wheel install `ls pygdf-*.whl` --force
sudo chmod -R a+rx /opt/h2oai/dai/python/
```

4a) Install mapd servers

```
# https://www.mapd.com/docs/v3.1.3/getting-started/installation/
sudo apt install default-jre-headless
export LD_LIBRARY_PATH=/usr/lib/jvm/default-java/jre/lib/amd64/server:$LD_LIBRARY_PATH # can be added to env, like to end of ~/.bashrc file
# Go to here and choose install needed.  The instructions are nicely setup, but look out for the typos.
https://www.mapd.com/platform/download-community/
# typo: sudo apt update sudo apt install mapd -> sudo apt update && sudo apt install mapd
# typo: cd $MAPD_PATH/systemd sudo ./install_mapd_systemd.sh -> cd $MAPD_PATH/systemd && sudo ./install_mapd_systemd.sh
# when following install_mapd_systemd.sh, just hit enter to accept all defaults (root as who runs, and ensure ~/.bashrc has correct MAPD_USER and MAPD_GROUP as root
# Once reach "Activation" step, change slightly what one does: sudo $MAPD_PATH/insert_sample_data -> cd $MAPD_PATH ; sudo $MAPD_PATH/insert_sample_data
```

or 4b) Install mapd servers from open-source repo:
```
https://github.com/mapd/mapd-core
```

5) Install mapd for python

```
https://arrow.apache.org/docs/python/development.html#development # but uses conda
/opt/h2oai/dai/dai-env.sh pip install pyarrow
/opt/h2oai/dai/dai-env.sh pip install pymapd # needs libraries like arrow and arrow_python, which above arrow webpage says how to install everything from source but that requires conda.  Stuck?  I just need the libs, not conda, so annoying.
```

```
    pip install pymapd
```
```
    creating build/temp.linux-x86_64-3.6/pymapd
    gcc-4.9 -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/home/jon/.pyenv/versions/3.6.4/lib/python3.6/site-packages/numpy/core/include -I/home/jon/.pyenv/versions/3.6.4/lib/python3.6/site-packages/pyarrow/include -I/home/jon/.pyenv/versions/3.6.4/include/python3.6m -c pymapd/shm.cpp -o build/temp.linux-x86_64-3.6/pymapd/shm.o -std=c++11
    cc1plus: warning: command line option ‘-Wstrict-prototypes’ is valid for C/ObjC but not for C++
    In file included from /home/jon/.pyenv/versions/3.6.4/lib/python3.6/site-packages/numpy/core/include/numpy/ndarraytypes.h:1816:0,
                     from /home/jon/.pyenv/versions/3.6.4/lib/python3.6/site-packages/numpy/core/include/numpy/ndarrayobject.h:18,
                     from /home/jon/.pyenv/versions/3.6.4/lib/python3.6/site-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from pymapd/shm.cpp:612:
    /home/jon/.pyenv/versions/3.6.4/lib/python3.6/site-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    g++-4.9 -pthread -shared -L/home/jon/.pyenv/versions/3.6.4/lib -Wl,-rpath=/home/jon/.pyenv/versions/3.6.4/lib -L/home/jon/.pyenv/versions/3.6.4/lib -Wl,-rpath=/home/jon/.pyenv/versions/3.6.4/lib build/temp.linux-x86_64-3.6/pymapd/shm.o -L/home/jon/.pyenv/versions/3.6.4/lib -larrow -larrow_python -lpython3.6m -o build/lib.linux-x86_64-3.6/pymapd/shm.cpython-36m-x86_64-linux-gnu.so -std=c++11
    /usr/bin/ld: cannot find -larrow
    /usr/bin/ld: cannot find -larrow_python
    collect2: error: ld returned 1 exit status
    error: command 'g++-4.9' failed with exit status 1
```

```
/opt/h2oai/dai/dai-env.sh pip install pymapd
```

```
  gcc-4.9 -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -Wstrict-prototypes -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -pipe -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -pipe -fPIC -I/opt/h2oai/dai/python/lib/python3.6/site-packages/numpy/core/include -I/opt/h2oai/dai/python/lib/python3.6/site-packages/pyarrow-0.8.0-py3.6-linux-x86_64.egg/pyarrow/include -I/opt/h2oai/dai/python/include/python3.6m -c pymapd/shm.cpp -o build/temp.linux-x86_64-3.6/pymapd/shm.o -std=c++11
  gcc-4.9: error: unrecognized command line option ‘-fno-plt’
  gcc-4.9: error: unrecognized command line option ‘-fno-plt’
  error: command 'gcc-4.9' failed with exit status 1

  ----------------------------------------

```


```
unset CC
unset CXX
/opt/h2oai/dai/dai-env.sh pip install pymapd
```

```
  x86_64-conda_cos6-linux-gnu-gcc -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -Wstrict-prototypes -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -pipe -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -pipe -fPIC -I/opt/h2oai/dai/python/lib/python3.6/site-packages/numpy/core/include -I/opt/h2oai/dai/python/lib/python3.6/site-packages/pyarrow-0.8.0-py3.6-linux-x86_64.egg/pyarrow/include -I/opt/h2oai/dai/python/include/python3.6m -c pymapd/shm.cpp -o build/temp.linux-x86_64-3.6/pymapd/shm.o -std=c++11
  unable to execute 'x86_64-conda_cos6-linux-gnu-gcc': No such file or directory
  error: command 'x86_64-conda_cos6-linux-gnu-gcc' failed with exit status 1

  -----
```

6) Smoke test

```
/opt/h2oai/dai/dai-env.sh python
import h2o4gpu
import pygdf
```

7) Notebook test

```
emacs -nw ~/.local/./share/jupyter/kernels/python3/kernel.json # and edit so python (just after argv line) is instead /opt/h2oai/dai/python/bin/python and edit display name to "python (dai)" to ensure see this name in jupyter notebook
cd ~/h2o4gpu/examples/py/goai/
/opt/h2oai/dai/dai-env.sh /opt/h2oai/dai/python/bin/jupyter notebook
```

