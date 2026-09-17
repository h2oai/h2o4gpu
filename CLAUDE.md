# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

H2O4GPU is a collection of GPU-accelerated machine-learning solvers (KMeans, PCA, Truncated SVD, GLM/ElasticNet via POGS, ARIMA, plus bundled XGBoost) with Python and R APIs. The Python package is a drop-in replacement for scikit-learn (`import h2o4gpu as sklearn`): it overrides selected sklearn classes with GPU implementations and falls back to sklearn's CPU code when a GPU option isn't supported.

It is consumed by **Driverless AI (DAI)** as a build dependency. DAI imports it directly — `from h2o4gpu.solvers import TruncatedSVD`, `from h2o4gpu.libs.lib_utils import GPUlib/CPUlib`, and `h2o4gpu.util.metrics`. When changing public solver APIs, the `libs` loader, or `util.metrics`, check for breakage in DAI's `h2oaicore` (transformers, systemutils, metrics).

## Architecture

Layered **C++/CUDA → C ABI → SWIG → Python**; understand the flow before touching a solver:

1. **CUDA/C++ kernels** — `src/gpu/` (kmeans, pca, tsvd, arima, glm/POGS) and `src/cpu/`. Shared headers in `src/include/`, `src/common/`.
2. **C ABI** — `src/interface_c/` (`h2o4gpu_c_api.h`): a flat C API over the C++ solvers for SWIG to bind.
3. **SWIG** — `src/swig/` (`ch2o4gpu_cpu.i`, `ch2o4gpu_gpu.i`) generates `_ch2o4gpu_cpu.so` / `_ch2o4gpu_gpu.so`, imported lazily via `h2o4gpu/libs/lib_utils.py` (`CPUlib.get()` / `GPUlib.get()`), which return `None` instead of raising when the `.so` is missing — so GPU-less environments degrade gracefully.
4. **Python solvers** — `src/interface_py/h2o4gpu/solvers/`: sklearn-style wrappers (e.g. `TruncatedSVDH2O`) that call the SWIG module and pick GPU vs CPU at runtime.

### scikit-learn overlay + generated `__init__.py`

h2o4gpu becomes a sklearn drop-in by **overlaying scikit-learn's source into its own namespace**. `scripts/prepare_sklearn.sh` clones the pinned scikit-learn (`SKLEARN_VERSION`, default 1.5.2 — keep in sync with `requirements_buildonly.txt`), renames `sklearn`→`h2o4gpu`, and `make apply-sklearn` builds it and symlinks the modules into `src/interface_py/h2o4gpu/`. **`make apply-sklearn` must run before `make py`**, or the package is missing `h2o4gpu.linear_model`, `h2o4gpu.cluster`, etc. (the overlaid modules are gitignored build artifacts).

`h2o4gpu/__init__.py` is **generated at build time** (don't hand-edit) by `scripts/apply_sklearn_initmerge.sh` from: `build_info.txt` + the overlaid sklearn `__init__.py` + `h2o4gpu/__init__.base.py` (which **overrides** the sklearn names). Edit `__init__.base.py` to change H2O4GPU's public exports.

### Bundled submodules

The wheel bundles H2O.ai forks (`.gitmodules`): `xgboost`/`xgboost_prev`, `LightGBM`, `py3nvml`, `nccl`, `cub`. Installing the wheel **overwrites** the env's `xgboost`/`py3nvml`/`lightgbm`, so keep these fork versions aligned with whatever shares the environment (e.g. DAI). Clone with `--recursive`. LightGBM is optional: it's only bundled if built (`make lightgbm_gpu` / `lightgbm_cpu`, which need boost-from-source) and is otherwise omitted from the wheel.

## Building

### Toolchain (CUDA 12.8 / Blackwell)

Requires the **CUDA 12.8+** toolkit, **gcc-toolset-13** (C++17), **CMake ≥ 3.18**, SWIG, OpenBLAS, **Python 3.11**, on a **glibc-2.28** base (Rocky/Alma 8) for broad compatibility. The compute-capability ladder is Pascal→Hopper plus Blackwell `sm_100`/`sm_120` (gated on CUDA ≥ 12.8); Kepler/Maxwell are dropped on CUDA 12.

### Recommended: cached Docker build

`Dockerfile.build-cuda12` bakes the toolchain + Python build deps into a cached image, so only the compile runs each time (the image cache rebuilds only when the Dockerfile or `requirements_buildonly.txt` change):

```bash
make docker-build-image                              # one-time: build/cache the builder image
make docker-build-cuda12                             # compile (default DOCKER_MAKE_TARGET=cpp, full arch ladder)
make docker-build-cuda12 DOCKER_MAKE_TARGET="cpp py" # also build the wheel -> src/interface_py/dist/
```

Runs as your uid:gid (artifacts are user-owned). Knobs: `DOCKER_MAKE_TARGET`, `DOCKER_USER` (set empty for container-root, e.g. on rootless Docker), `DOCKER_RUN_FLAGS` (e.g. `--gpus all` for GPU tests), `CUDA_IMAGE`. `make docker-build-shell` opens a shell in the image with the tree mounted.

### Native make targets (run inside the build env)

```bash
make apply-sklearn   # build the scikit-learn overlay (REQUIRED before `make py`)
make cpp             # compile the C++/CUDA .so (full arch ladder)
make py              # build the wheel
make install         # pip-install the built wheel
make build install   # == make cpp py install
```

Avoid `make fullinstall`: it runs `clean` (which resets the xgboost submodules via `git submodule update`) and does **not** run `apply-sklearn`.

**Flags:** `DEV_BUILD=ON` (single arch sm_61 — much faster compile), `CMAKE_BUILD_TYPE=Debug`, `USENCCL=0`, `USENVTX=ON`.

**Mixed-arch gotcha:** `make cpp` does not clean `build/`. Switching between `DEV_BUILD` and a full build over the same `build/` yields a broken `.so` (undefined Thrust symbols — Thrust's inline namespace encodes the arch set). Run `make clean_cpp` (or `rm -rf build`) when changing arch flags.

### Wheel version / release

`BASE_VERSION` lives in `make/version.mk`; the wheel version is `BASE_VERSION` + `H2O4GPU_SUFFIX`. The default suffix is `+local_<git-describe>` (dev). For a clean release version, pass an empty suffix:

```bash
make docker-build-cuda12 DOCKER_MAKE_TARGET="cpp py H2O4GPU_SUFFIX= H2O4GPU_BUILD=release"
# -> src/interface_py/dist/h2o4gpu-<BASE_VERSION>-cp311-cp311-linux_x86_64.whl
```

## Testing

```bash
make test            # build, then run the Python suite (tests/python/{small,big,open_data,xgboost})
make test_cpp        # C++ googletest suite (tests/cpp/)
```

Tests run under pytest (xdist `-n`, `--timeout`) and are marked `multi_gpu` vs `not multi_gpu`. Single test: `pytest -v -s tests/python/open_data/kmeans/test_kmeans.py`. Many tests need data first: `make sync_open_data` (pulls from S3).

## Code style

- **C/C++/CUDA**: Google style (`.clang-format`: `BasedOnStyle: Google`, `IndentCaseLabels: true`, `NamespaceIndentation: None`).
- **Python**: Google style. Lint: `cd src/interface_py && make pylint`; format: `make pyformat` (yapf) — skips the generated `__init__.py`.

## Adding a solver

`EXAMPLE_SOLVER.md` walks the full path: C++ kernel → C API → SWIG `.i` → Python wrapper → `__init__.base.py` export → tests.
