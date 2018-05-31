# H2O4GPU

[![Join the chat at https://gitter.im/h2oai/h2o4gpu](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/h2oai/h2o4gpu)

## Troubleshooting

#### Problems with numpy version mismatch ###

Good idea to check if duplicate python packages installed

```
pip freeze

```

and pip uninstall any prior version you had and pip install the
version we tried to install.  E.g. on conda you might need to do:

```
pip uninstall numpy
pip install numpy==1.13.1 # or whatever version was attempted to be installed by the wheel
```

### h2o4gpu not found ###

After pip installing the wheel, make sure you use a fresh bash
environment to ensure the python cache is not used.

### Can't properly load ch2o4gpu_gpu.so ###

e.g. Warning: `h2o4gpu_kmeans_lib shared object (dynamic library) ~/h2o4gpu/lib/python3.6/site-packages/h2o4gpu/libs/../../ch2o4gpu_gpu.so failed to load.`

This can be cause by several issues:

1) some other library is missing. One can run:

```
cd <your python environment path>/site-packages/
#e.g. for pyenv: /home/$USER/.pyenv/versions/3.6.1/lib/python3.6/site-packages/
ldd ch2o4gpu_gpu.so
```

To see if things are missing.

2) Make sure you installed cuda and it's linked correctly to /usr/local/cuda. And make sure you set the environment variables related to cuda.

3) Run `ldd --version`, we currently require version `2.23` or higher. If your system is running a lower version please update if possible or build the project yourself on your machine.

4) Make sure you are running CUDA 8.x or CUDA 9.x.

5) If compiled with icc (default if present) and have conda, need to do:

```
conda install --no-dep -c intel icc_rt
```
