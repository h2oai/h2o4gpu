# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

H2O4GPU is a collection of GPU-accelerated machine learning solvers with Python and R APIs. The Python package is a drop-in replacement for scikit-learn (`import h2o4gpu as sklearn`): it overrides selected sklearn classes with GPU implementations and silently falls back to sklearn's CPU algorithms when a GPU option isn't supported.

This package is consumed by **Driverless AI (DAI)** as a build dependency. DAI imports it directly — e.g. `from h2o4gpu.solvers import TruncatedSVD`, `from h2o4gpu.libs.lib_utils import GPUlib/CPUlib`, and `h2o4gpu.util.metrics`. When changing public solver APIs, the `libs` loader, or `util.metrics`, check for breakage in DAI's consuming modules (its `h2oaicore` transformers, systemutils, and metrics).

## Architecture

The code is layered C++/CUDA → C ABI → SWIG → Python. Understanding this flow is essential before touching a solver:

1. **CUDA/C++ kernels** — `src/gpu/` (GPU solvers: kmeans, pca, tsvd, factorization, arima, glm) and `src/cpu/` (CPU equivalents). Shared headers in `src/include/` and `src/common/`.
2. **C ABI** — `src/interface_c/` (`h2o4gpu_c_api.h`) exposes a flat C API over the C++ solvers so SWIG can bind it.
3. **SWIG** — `src/swig/` (`ch2o4gpu_cpu.i`, `ch2o4gpu_gpu.i`) generates two Python extension modules: `_ch2o4gpu_cpu.so` and `_ch2o4gpu_gpu.so`. They are imported lazily via `h2o4gpu/libs/lib_utils.py` (`CPUlib.get()` / `GPUlib.get()`), which returns `None` rather than raising when the shared object is missing — so GPU-less environments degrade gracefully.
4. **Python solvers** — `src/interface_py/h2o4gpu/solvers/` contains sklearn-style wrapper classes (e.g. `KMeansH2O` in `kmeans.py`) that call the loaded SWIG module. Each picks GPU vs CPU at runtime.

### The generated `__init__.py` (important gotcha)

`h2o4gpu/__init__.py` is **generated at build time** — do not hand-edit it. It is assembled by `scripts/apply_sklearn_initmerge.sh` from three sources, concatenated in order:
- `src/interface_py/build_info.txt`
- the upstream sklearn `__init__.py` (with `__version__` stripped) — this is how all sklearn names become importable from `h2o4gpu`
- `src/interface_py/h2o4gpu/__init__.base.py` — H2O4GPU's own exports, which **override** the sklearn names pulled in above

Edit `__init__.base.py` to change H2O4GPU's public exports. The `make clean` step truncates `__init__.py` to empty.

### Bundled third-party packages

The wheel bundles several H2O.ai forks pulled in as git submodules (`.gitmodules`): `xgboost` (and `xgboost_prev`), `LightGBM`, `py3nvml`, `nccl`, `cub`. The Python build symlinks these into `src/interface_py/` so they ship inside the h2o4gpu wheel. Installing the wheel overwrites the user's `xgboost`/`py3nvml`/`lightgbm` with these validated versions. Always clone with `--recursive` and run `git submodule update` after pulling.

## Build & install

Requires Linux, CUDA at `/usr/local/cuda` (`CUDA_HOME`), GCC 4.9+, CMake, SWIG, OpenBLAS. See `DEVEL.md` for full environment setup (boost-from-source for static LightGBM linking, etc.).

```bash
make fullinstall        # clean + submodules + deps + full build + pip install the wheel (slow; the safe default)
```

Incremental rebuilds (use these during development instead of `fullinstall`):

```bash
make cpp                # only C++/CUDA changed
make py                 # only Python changed (rebuilds the wheel)
make install            # only packaging changed (reinstalls the wheel)
make build install      # C++ changed and you want it repackaged + installed (== make cpp py install)
```

Useful build flags (see `make/config.mk`):
- `DEV_BUILD=ON` — build a single CUDA compute capability (6.1); much faster compiles.
- `CMAKE_BUILD_TYPE=Debug` — debug build.
- `USENCCL=0` — disable NCCL (only XGBoost uses it).
- `USENVTX=ON` — enable nvToolsExt profiling markers.

Version number lives in `make/version.mk` (`BASE_VERSION`).

## Testing

```bash
make test               # build_quick + run the Python test suite
make testquick          # run tests without rebuilding (dotest)
make dotestfast         # the four fast smoke tests (glm/xgb/tsvd/kmeans)
make testperf           # performance + accuracy benchmarks vs h2o-3
make test_cpp           # C++ googletest suite (tests/cpp/)
```

Python tests live in `tests/python/{small,big,open_data,xgboost}/` and run under pytest with xdist (`-n`) and `--timeout`. Tests are marked `multi_gpu` vs `not multi_gpu`. To run a single test directly:

```bash
pytest --verbose -s tests/python/open_data/kmeans/test_kmeans.py
```

Many tests need data fetched first via `make sync_open_data` / `sync_small_data` / `sync_other_data` (these pull from S3). `sh tests/showresults.sh` summarizes errors and timings after a perf run.

## Code style

- **C/C++/CUDA**: Google C++ Style. `.clang-format` is `BasedOnStyle: Google` with `IndentCaseLabels: true`, `NamespaceIndentation: None`.
- **Python**: Google Python Style. Lint with `cd src/interface_py && make pylint` (uses `tools/pylintrc`). Auto-format with `make pyformat` (yapf, google style) — note this deliberately skips the generated `__init__.py`.

## Adding a new solver

`EXAMPLE_SOLVER.md` walks through adding a solver end-to-end across all layers (C++ kernel → C API → SWIG `.i` → Python wrapper → `__init__.base.py` export → tests).
