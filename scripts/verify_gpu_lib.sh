#!/usr/bin/env bash
#
# Verify that a compiled _ch2o4gpu_gpu.so is still intact after
# `strip --strip-unneeded`. This is a structural check that needs only the CUDA
# toolkit (cuobjdump) + binutils (nm, readelf) -- NO GPU required -- so it can
# run as part of the build and fail loudly if strip ever damages the library
# instead of letting a broken .so ship and only break in production.
#
# It asserts only what strip can affect:
#   1. the .so is actually stripped (no .debug_* sections),
#   2. the CUDA fatbin survived: SASS cubins are still enumerable (if strip had
#      corrupted .nv_fatbin, cuobjdump would list none),
#   3. the CPython init symbol is still exported (else `import` fails).
#
# It deliberately does NOT check the PTX / arch SET here -- that is a build-config
# concern (did we compile compute_120?), not a strip-integrity one, and belongs
# in the post-build wheel autopsy (`cuobjdump --list-ptx`). Strip never removes
# PTX selectively, so folding it in only risks false build failures.
#
# Usage: verify_gpu_lib.sh <path-to-_ch2o4gpu_gpu.so>
#
# NB: no `pipefail` -- `grep -q` on the large `nm -D` stream closes the pipe
# early and SIGPIPEs the upstream command, which under pipefail would report a
# false failure on a *matching* symbol. The explicit checks below handle real
# failures.
set -u

so="${1:-}"
if [ -z "$so" ]; then
    echo "usage: $0 <path-to-_ch2o4gpu_gpu.so>" >&2
    exit 2
fi
if [ ! -f "$so" ]; then
    echo "FATAL: $so does not exist" >&2
    exit 1
fi

fail() { echo "FATAL: $1" >&2; exit 1; }

# 1. Must be stripped.
if readelf -S "$so" 2>/dev/null | grep -q '\.debug_'; then
    fail "$so still contains .debug_* sections (strip did not run?)"
fi

# 2. Fatbin / SASS cubins must still be readable.
archs=$(cuobjdump --list-elf "$so" 2>/dev/null | grep -oE 'sm_[0-9]+' | sort -u)
if [ -z "$archs" ]; then
    fail "no SASS cubins found in $so after strip -- device code lost!"
fi
echo "OK: SASS cubins intact after strip: $(echo $archs | tr '\n' ' ')"

# 3. Python module init symbol must survive (else the module is not importable).
if ! nm -D "$so" 2>/dev/null | grep -q 'PyInit__ch2o4gpu_gpu'; then
    fail "PyInit__ch2o4gpu_gpu not exported in $so -- module not importable!"
fi
echo "OK: PyInit__ch2o4gpu_gpu exported (module is importable)."

echo "PASS: $so survived strip (fatbin + init symbol intact)."
