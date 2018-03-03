# H2O4GPU Installation on Amazon Web Services (AWS)

This describes how to install H2O4GPU on a fresh AWS Ubuntu 16.0.4 GPU instance (tested on g3.4x, which has one Tesla M60).

### NVIDIA CUDA:

Dependencies:

```
sudo apt-get update -y
sudo apt-get install -y gcc
sudo apt-get install -y make
```

NVIDIA CUDA 9.0 — check [here](https://developer.nvidia.com/cuda-downloads) for up-to-date links (note that, as of 03/01/18 H2O4GPU does not support CUDA 9.1).

```
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
sudo sh cuda_9.0.176_384.81_linux-run
```
Agree to terms (press SPACE to page down) and install with all defaults.

Verify drivers installed:

```
nvidia-smi
```

Environment variables:

```
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64/:$CUDA_HOME/lib/:$CUDA_HOME/extras/CUPTI/lib64
```

### Python

To install Python3.6 via apt-get on Ubuntu 16.0.4:

```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update -y
sudo apt-get install -y python3.6
sudo rm /usr/bin/python3
sudo ln -s /usr/bin/python3.6 /usr/bin/python3
sudo apt-get install -y python3.6-dev
sudo apt-get install -y python3-pip
```
### H2O4GPU

Dependencies:

```
sudo apt-get install libopenblas-dev pbzip2
```

Get and install python package:

```
wget https://s3.amazonaws.com/h2o-release/h2o4gpu/releases/stable/ai/h2o/h2o4gpu/0.1-nccl-cuda9/h2o4gpu-0.1.0-py36-none-any.whl
pip3 install h2o4gpu-0.1.0-py36-none-any.whl
```

Try H2O4GPU:

```
python3
>>> import h2o4gpu
>>> import numpy as np
>>> X = np.array([[1.,1.], [1.,4.], [1.,0.]])
>>> model = h2o4gpu.KMeans(n_clusters=2,random_state=1234).fit(X)
>>> model.cluster_centers_
array([[1. , 0.5],
       [1. , 4. ]])
```
