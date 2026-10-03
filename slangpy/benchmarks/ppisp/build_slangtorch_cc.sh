#!/bin/bash
# Build the slangtorch PPISP extension with NRE-style optimized flags.
#
# This recompiles the JIT-generated .cu and .cpp with -O3 and --as-needed,
# producing a smaller and faster .so (12us dispatch vs 31us from JIT).
#
# Prerequisites:
#   - slangtorch installed (pip install slangtorch)
#   - The benchmark must have been run at least once to JIT-generate the sources
#   - CUDA toolkit available
#
# Usage:
#   ./build_slangtorch_cc.sh [python_path]
#
# Example:
#   ./build_slangtorch_cc.sh /builds/nre/.venv_bench/bin/python

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${1:-python}"

# Find the JIT-generated sources
CACHE_DIR=$(find "$SCRIPT_DIR/.slangtorch_cache/ppisp_slangtorch" -name "ppisp_slangtorch_cuda.cu" -printf "%h" -quit 2>/dev/null)
if [ -z "$CACHE_DIR" ]; then
    echo "No JIT-generated sources found. Running benchmark once to generate them..."
    cd /tmp && "$PYTHON" -c "
import torch, slangtorch, os
m = slangtorch.loadModule(
    os.path.join('$SCRIPT_DIR', 'ppisp_slangtorch.slang'),
    verbose=True,
    defines={'NUM_VIGNETTING_ALPHA_TERMS': '3'},
)
print('Module loaded, sources generated.')
"
    CACHE_DIR=$(find "$SCRIPT_DIR/.slangtorch_cache/ppisp_slangtorch" -name "ppisp_slangtorch_cuda.cu" -printf "%h" -quit)
    if [ -z "$CACHE_DIR" ]; then
        echo "ERROR: Still can't find generated sources."
        exit 1
    fi
fi

echo "Source dir: $CACHE_DIR"

# Get paths from Python/torch
TORCH_DIR=$("$PYTHON" -c "import torch, os; print(os.path.dirname(torch.__file__))")
TORCH_LIB="$TORCH_DIR/lib"
TORCH_INC="$TORCH_DIR/include"
TORCH_INC_CSRC="$TORCH_DIR/include/torch/csrc/api/include"
PYTHON_INC=$("$PYTHON" -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_LIB=$("$PYTHON" -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
PYTHON_LDLIB=$("$PYTHON" -c "import sysconfig; v = sysconfig.get_config_var('LDLIBRARY'); print(v.replace('lib','').replace('.so','').replace('.dylib',''))")
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ ! -d "$CUDA_HOME" ]; then
    CUDA_HOME=$("$PYTHON" -c "import torch; print(torch.utils.cpp_extension.CUDA_HOME or '/usr/local/cuda')" 2>/dev/null || echo "/usr/local/cuda")
fi

BUILD_DIR=$(mktemp -d)
OUTPUT="$SCRIPT_DIR/ppisp_slangtorch_cc.so"

echo "Torch: $TORCH_DIR"
echo "CUDA:  $CUDA_HOME"
echo "Build: $BUILD_DIR"

# Step 1: Compile CUDA
echo "=== Compiling CUDA ==="
"$CUDA_HOME/bin/nvcc" \
    -c "$CACHE_DIR/ppisp_slangtorch_cuda.cu" \
    -o "$BUILD_DIR/ppisp_slangtorch_cuda.o" \
    -x cu \
    -O3 \
    --use_fast_math \
    --generate-line-info \
    -D__CUDA_NO_HALF_OPERATORS__ \
    -D__CUDA_NO_HALF_CONVERSIONS__ \
    -D__CUDA_NO_BFLOAT16_CONVERSIONS__ \
    -D__CUDA_NO_HALF2_OPERATORS__ \
    -w \
    -Xcompiler -fPIC \
    -I"$TORCH_INC" \
    -I"$TORCH_INC_CSRC" \
    -I"$PYTHON_INC" \
    -I"$CUDA_HOME/include" \
    -std=c++17

# Step 2: Compile C++ binding
echo "=== Compiling C++ binding ==="
g++ \
    -c "$CACHE_DIR/ppisp_slangtorch.cpp" \
    -o "$BUILD_DIR/ppisp_slangtorch.o" \
    -fPIC \
    -O3 \
    -w \
    -DTORCH_API_INCLUDE_EXTENSION_H \
    -DTORCH_EXTENSION_NAME=libppisp_slangtorch_cc \
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    -I"$TORCH_INC" \
    -I"$TORCH_INC_CSRC" \
    -I"$PYTHON_INC" \
    -I"$CUDA_HOME/include" \
    -std=c++17

# Step 3: Link
echo "=== Linking ==="
g++ \
    -shared \
    "$BUILD_DIR/ppisp_slangtorch.o" \
    "$BUILD_DIR/ppisp_slangtorch_cuda.o" \
    -o "$OUTPUT" \
    -Wl,--no-undefined \
    -Wl,--as-needed \
    -L"$TORCH_LIB" -lc10 -lc10_cuda -ltorch_cpu -ltorch_python -ltorch \
    -L"$CUDA_HOME/lib64" -lcudart \
    -L"$PYTHON_LIB" -l"$PYTHON_LDLIB"

rm -rf "$BUILD_DIR"

echo "=== Done ==="
ls -lh "$OUTPUT"
echo ""
echo "Run benchmark with:"
echo "  cd /tmp && $PYTHON -m pytest $SCRIPT_DIR/../test_benchmark_ppisp.py -v -s -k '(slangpy or slangtorch) and not pytorch and 1000000 and forward and not gpu'"
