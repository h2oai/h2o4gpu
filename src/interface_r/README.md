# h2o4gpu: R Interface to H2O4GPU

[![CRAN\_Status\_Badge](https://www.r-pkg.org/badges/version/h2o4gpu)](https://cran.r-project.org/package=h2o4gpu)

This directory contains the R package for **H2O4GPU**, a collection of GPU solvers by [H2O.ai](https://www.h2o.ai/) with APIs in Python and R.  The Python API builds upon the easy-to-use [scikit-learn](https://scikit-learn.org) API.  The **h2o4gpu** R package is a wrapper around the **h2o4gpu** Python module.  The R package makes use of RStudio's [reticulate](https://rstudio.github.io/reticulate/) package for facilitating access to Python libraries through R.

## Installation

There are a few [system requirements](https://github.com/h2oai/h2o4gpu#requirements), including Ubuntu 16.04+, Python >=3.6, R >=3.1, CUDA 8 or 9, and a machine with Nvidia GPUs.  The code should still run if you have CPUs, but it will fall back to scikit-learn CPU based versions of the algorithms.

The **h2o4gpu** Python module is a prerequisite for the R package. So first, follow the instructions [here](https://github.com/h2oai/h2o4gpu#user-installation) to install the **h2o4gpu** Python package (either at the system level or in a Python virtual envivonment). The easiest thing to do is to `pip install` the stable release `whl` file. To ensure compatibility, the Python package version number should match the R package version number. 

The recomended way of installing the R package can is from CRAN using `install.packages("h2o4gpu")`. To install the development version of the **h2o4gpu** R package, you can install directly from GitHub as follows:

``` r
library(devtools)
devtools::install_github("h2oai/h2o4gpu", subdir = "src/interface_r")
```


### Virtual environments

Using a Python [virtual environment](https://packaging.python.org/tutorials/installing-packages/#creating-virtual-environments) is a good solution if you don't want to upgrade your main Python installation to 3.6.  If you installed the **h2o4gpu** Python module into a virtual environment, you will have to add a line of code to tell R which Python envivonment you want to use:

``` r
library(reticulate)
use_virtualenv("/home/username/venv/h2o4gpu")  # set this to the path of your venv
```
If you have installed **h2o4gpu** Python module using Anaconda, then you can use the `use_condaenv()` function instead.  More information about Python environment configuration is available in the reticulate [user guide](https://rstudio.github.io/reticulate/articles/versions.html).


### Test installation

To test your installation, try the following example that builds a simple random forest classifier:

``` r
library(h2o4gpu)

# Prepare data
x <- iris[1:4]
y <- as.integer(iris$Species) # all columns, including the response, must be numeric

# Initialize and train the classifier
model <- h2o4gpu.random_forest_classifier() %>% fit(x, y)

# Make predictions
pred <- model %>% predict(x)
```

For examples of how to use all of the functions in the package, please visit the vignettes section [here](https://cran.r-project.org/package=h2o4gpu).


## Troubleshooting

If you have any issues, or have any recommendations to the installation instructions, please let us know by filing a GitHub issue.  If there are installation issues, the first thing to check is the [system requirements](https://github.com/h2oai/h2o4gpu#requirements).


### GPUs

The first thing to check is that you actually have a machine with Nvidia GPUs.  If you can run the `nvidia-smi` command in the shell and get an output that looks similar to this, it means you indeed have GPUs and working drivers on your machine:

```
username@gpubox:~$ nvidia-smi
Tue Mar 27 11:38:14 2018
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 387.34                 Driver Version: 387.34                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 1080    On   | 00000000:02:00.0 Off |                  N/A |
| 27%   39C    P8    10W / 180W |     12MiB /  8112MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX 1080    On   | 00000000:81:00.0 Off |                  N/A |
| 27%   39C    P8    11W / 180W |     12MiB /  8114MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```



### CUDA 

To check if CUDA 8 or 9 is installed, run the `nvcc --version` command.  If you see this:

```
-bash: nvcc: command not found
```
That means that CUDA is not installed.  If you do have CUDA installed, you will see something like this:

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2016 NVIDIA Corporation
Built on Tue_Jan_10_13:22:03_CST_2017
Cuda compilation tools, release 8.0, V8.0.61
```


### Python issues

If you try to train a model and you get a non-descript error like this:

```
> # Initialize and train the classifier
> model <- h2o4gpu.random_forest_classifier() %>% fit(x, y)
Error: 
```

Or if you receive a full error like this:

```
Error: Python module h2o4gpu was not found.

Detected Python configuration:

python:         /usr/local/bin/python
libpython:      /usr/local/Cellar/python/2.7.10/Frameworks/Python.framework/Versions/2.7/lib/python2.7/config/libpython2.7.dylib
pythonhome:     /usr/local/Cellar/python/2.7.10/Frameworks/Python.framework/Versions/2.7:/usr/local/Cellar/python/2.7.10/Frameworks/Python.framework/Versions/2.7
version:        2.7.10 (default, Jun  1 2015, 09:44:56)  [GCC 4.2.1 Compatible Apple LLVM 6.1.0 (clang-602.0.53)]
numpy:          /usr/local/lib/python2.7/site-packages/numpy
numpy_version:  1.12.0
h2o4gpu:        [NOT FOUND]

python versions found: 
 /usr/local/bin/python
 /usr/bin/python
 /usr/local/bin/python3
```

That means that R package cannot locate the **h2o4gpu** Python module.  To fix this, make sure you have installed the **h2o4gpu** Python module, and that you are using one of the **reticulate** functions (e.g. `use_python()`, `use_virtualenv()`, `use_condaenv()`) to specify which Python environment you want to use.

If you have multiple versions of Python installed on your machine and don't want to use the primary version (the one you get when you type `python` at the command line), then you may consider using `reticulate::use_python()` function to explicitly specify which one to use:

``` r
library(reticulate)
use_python("/usr/local/bin/python")
```

If you encounter an issue that is not documented here, please file a GitHub issue to tell us about it.
