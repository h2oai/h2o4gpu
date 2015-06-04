#### To install
In R execute:
```
> setwd("<path-to-pogs-root>/src/interface_r/")
> install.packages("pogs_1.0.tar.gz", repos = NULL, type="source")
```
For the GPU version replace the second line with
```
> install.packages("pogs_1.0_gpu.tar.gz", repos = NULL, type="source")
```
Hint: Use `getwd()` to get the current directory.

#### To build from scratch
1. In your terminal, navigate to `$ cd <path-to-pogs-root>/src/interface_r/pogs`
   and execute `./cfg`.
2. Modify `<path-to-pogs-root>/src/interface_r/pogs/src/config.mk` by settingx
   `TARGET=[cpu/gpu]`, to target the respective platform.
   Also update `CUDA_HOME` if necessary.
3. In your terminal, navigate to `$ cd <path-to-pogs-root>/src/interface_r`
   and execute `R CMD build pogs`.
4. Install as specified above.

