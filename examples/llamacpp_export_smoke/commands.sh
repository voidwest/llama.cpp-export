#!/usr/bin/env bash
# llama.cpp layer export smoke — exact reproduction commands
# Run from the llama.cpp-export repository root.
set -euo pipefail

LLAMACPP_DIR="$(pwd)"
MODEL_DIR="${HOME}/ember"
BUILD_DIR="${LLAMACPP_DIR}/build"
OUT_DIR="/tmp/llamacpp_export_smoke"
DUMP_TOOL="${LLAMACPP_DIR}/dump_llamacpp_layers"

echo "=== Step 1: Build llama.cpp ==="
cmake -B "${BUILD_DIR}" \
    -DGGML_NATIVE=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target llama -j"$(nproc)"

echo "=== Step 2: Build dump_llamacpp_layers ==="
g++ -std=c++17 \
    -I"${LLAMACPP_DIR}/include" \
    -I"${LLAMACPP_DIR}/ggml/include" \
    -I"${LLAMACPP_DIR}/src" \
    "${LLAMACPP_DIR}/tools/dump_llamacpp_layers.cpp" \
    "${BUILD_DIR}/src/libllama.a" \
    "${BUILD_DIR}/ggml/src/libggml.a" \
    "${BUILD_DIR}/ggml/src/libggml-base.a" \
    "${BUILD_DIR}/ggml/src/libggml-cpu.a" \
    -lpthread -ldl -lm -fopenmp \
    -o "${DUMP_TOOL}"

mkdir -p "${OUT_DIR}"

echo "=== Step 3: Dump Qwen3-0.6B layers (BOS token) ==="
"${DUMP_TOOL}" \
    "${MODEL_DIR}/Qwen3-0.6B-Q8_0.gguf" \
    "" \
    "${OUT_DIR}/qwen3_0_6b_layers.bin" \
    16
echo "Expected: 28 layers x 1024 embd = 28672 floats (114688 bytes)"

echo "=== Step 4: Dump Qwen3-0.6B layers (prompt) ==="
"${DUMP_TOOL}" \
    "${MODEL_DIR}/Qwen3-0.6B-Q8_0.gguf" \
    "Hello world" \
    "${OUT_DIR}/qwen3_0_6b_hello_layers.bin" \
    16
echo "Expected: 28 layers x 1024 embd = 28672 floats (114688 bytes)"

echo "=== Step 5: Dump Llama-3.2-1B layers (prompt) ==="
"${DUMP_TOOL}" \
    "${MODEL_DIR}/Llama-3.2-1B-Instruct-Q8_0.gguf" \
    "Hello" \
    "${OUT_DIR}/llama3_2_1b_hello_layers.bin" \
    16
echo "Expected: 16 layers x 2048 embd = 32768 floats (131072 bytes)"

echo "=== Step 6: Determinism check (qwen3, same prompt twice) ==="
"${DUMP_TOOL}" \
    "${MODEL_DIR}/Qwen3-0.6B-Q8_0.gguf" \
    "Hello world" \
    "${OUT_DIR}/qwen3_run_a.bin" \
    16
"${DUMP_TOOL}" \
    "${MODEL_DIR}/Qwen3-0.6B-Q8_0.gguf" \
    "Hello world" \
    "${OUT_DIR}/qwen3_run_b.bin" \
    16

if cmp -s "${OUT_DIR}/qwen3_run_a.bin" "${OUT_DIR}/qwen3_run_b.bin"; then
    echo "PASS: deterministic (byte-identical)"
else
    echo "FAIL: non-deterministic output"
    exit 1
fi

echo "=== Step 7: gguf-parity-tools self-parity ==="
# Requires: pip install git+https://github.com/voidwest/gguf-parity-tools.git
gguf-parity compare-layers \
    --candidate "${OUT_DIR}/qwen3_0_6b_hello_layers.bin" \
    --reference "${OUT_DIR}/qwen3_0_6b_hello_layers.bin" \
    --layers 28 \
    --hidden-size 1024 \
    --out "${OUT_DIR}/qwen3_self_report"
echo "Expected: status=pass, cosine=1.0, max_abs_diff=0.0"

echo "=== Step 8: Verify file sizes ==="
python3 -c "
import os, numpy as np

checks = {
    'qwen3 BOS':  ('${OUT_DIR}/qwen3_0_6b_layers.bin',  28*1024),
    'qwen3 hello':('${OUT_DIR}/qwen3_0_6b_hello_layers.bin', 28*1024),
    'llama hello': ('${OUT_DIR}/llama3_2_1b_hello_layers.bin', 16*2048),
}
all_ok = True
for name, (path, expected) in checks.items():
    data = np.fromfile(path, dtype=np.float32)
    ok = len(data) == expected
    tag = 'PASS' if ok else 'FAIL'
    print(f'{tag}: {name}: {len(data)} floats (expected {expected})')
    if not ok:
        all_ok = False
if not all_ok:
    exit(1)
"

echo ""
echo "=== All smoke checks passed ==="
echo "Output files in: ${OUT_DIR}"
