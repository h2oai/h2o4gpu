#### To install
```
In your terminal, in this directory, execute: `R CMD INSTALL h2o4gpu_*XXX*.tgz`
```
where `XXX` is one of the following
- (empty) -- This package should work on any system, but needs compiling.
- `_gpu` -- This package requires nvcc installed and the environment variable `CUDA_HOME` to be set.
- `_osx` -- This is a pre-compiled binary, but will only work on OS X.
- `_gpu_osx` -- This is a pre-compiled binary, but will only work if you're on OS X
  and have a CUDA enabled GPU.

Hint: Use `getwd()` to get the current directory.

#### To build and install from scratch
1. Modify `<path-to-h2o4gpu-root>/src/interface_r/h2o4gpu/src/config.mk` by settingx
   `TARGET=[cpu/gpu]`, to target the respective platform.
   Also update `CUDA_HOME` if necessary.
2. In your terminal, navigate to `cd <path-to-h2o4gpu-root>/src/interface_r`
   and execute `R CMD INSTALL --build h2o4gpu` (or `R CMD build h2o4gpu` for
   a non-compiled package) to install the R package.
   For a parallel R build with all your cores, do: MAKE="make -j" R CMD INSTALL --build h2o4gpu



