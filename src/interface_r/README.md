### To install
```
setwd("<pogs>/src/interface_r/")
install.packages("pogs_1.0.tar.gz", repos = NULL, type="source")
```
Hint: Use `getwd()` to get the current directory.

### To build from scratch
1. cd pogs
2. ./cfg
3. In pogs/src/config.mk set `TARGET=[cpu/gpu]`, to target the
   respective platform. Also update `CUDA_HOME` if necessary.
4. From the current directory (interface_r) run `R CMD build pogs`.
5. Install as specified above.

