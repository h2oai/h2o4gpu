#### To install
In R execute:
```
> setwd("<path-to-pogs-root>/src/interface_r/")
> install.packages("pogs_1.0XXX.tgz", repos = NULL, type="source")
```
where `XXX` is one of the following
- (empty) -- This package should work on any system, but needs compiling.
- `_gpu` -- This package requires nvcc installed and the environment variable `CUDA_HOME` to be set.
- `_osx` -- This is pre-compiled binary, but will only work on OSX.
- `_gpu_osx` -- This is pre-compiled binary, but will only work if you're on OS X
  and have a CUDA enabled GPU.

Hint: Use `getwd()` to get the current directory.

#### To build from scratch
1. In your terminal, navigate to `cd <path-to-pogs-root>/src/interface_r/pogs`
   and execute `./cfg`.
2. Modify `<path-to-pogs-root>/src/interface_r/pogs/src/config.mk` by settingx
   `TARGET=[cpu/gpu]`, to target the respective platform.
   Also update `CUDA_HOME` if necessary.
3. In your terminal, navigate to `cd <path-to-pogs-root>/src/interface_r`
   and execute `R CMD build --binary pogs` (or `R CMD build pogs` for
   a non-compiled package).
4. Install as specified above.

