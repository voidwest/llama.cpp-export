# llama.cpp Layer Dumps

`gguf-parity compare-layers` compares flat `float32` hidden-state dumps. It does
not depend on llama.cpp directly, but llama.cpp is a useful reference for
producing trusted per-layer hidden states.

With this llama.cpp fork, per-layer hidden state dumping is a **built-in
feature** -- no manual patching required.

## Output Contract

The reference dump should be:

- native-endian `float32`
- flat row-major array
- shape `[layers, hidden_size]`
- one row per layer, layer 0 first
- vector for the final prompt token

Example metadata:

```json
{
  "engine": "llama.cpp",
  "layers": 35,
  "hidden_size": 1536,
  "prompt": "",
  "token_ids": [2],
  "capture_point": "block_output_after_residual"
}
```

## Helper

`tools/dump_llamacpp_layers.cpp` is a small helper that links against the
llama.cpp library and uses the built-in `llama_set_output_layer_out` +
`llama_save_output_layer_out` API. It requires no custom patches to the
llama.cpp source.

Build shape:

```bash
cd /path/to/llama.cpp
cmake -B build -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF
cmake --build build --target llama -j"$(nproc)"
g++ -std=c++17 -I./include -I./ggml/include -I./src \
  tools/dump_llamacpp_layers.cpp \
  ./build/src/libllama.a \
  ./build/ggml/src/libggml.a \
  ./build/ggml/src/libggml-base.a \
  ./build/ggml/src/libggml-cpu.a \
  -lpthread -ldl -lm -o dump_llamacpp_layers
```

Run shape:

```bash
./dump_llamacpp_layers model.gguf "" llama_layers.bin 16
```

Then compare:

```bash
gguf-parity compare-layers \
  --candidate engine_layers.bin \
  --reference llama_layers.bin \
  --layers 35 \
  --hidden-size 1536 \
  --out report
```

## API Usage (Programmatic)

For C/C++ consumers that want to dump layers without the helper:

```c
#include "llama.h"

// Enable layer output before decoding
llama_set_output_layer_out(ctx, true);

// Decode as usual
llama_decode(ctx, batch);

// Save to file
llama_save_output_layer_out(ctx, "layers.bin");

// Or access the raw buffer in memory:
//   n_layer = llama_model_n_layer(model);
//   n_embd  = llama_model_n_embd(model);
//   float * data = llama_get_embeddings_layer_out(ctx);
//   // data is flat f32 array of shape [n_layer * n_embd]
```

## Notes

- Layer dumps are captured at the per-layer block output point (`l_out` callback
  in the model graph), which occurs after the residual add and control-vector
  application. This is the same capture point used for all model architectures.
- The feature works across all model architectures in llama.cpp (125+ models).
- The `embd_layer_out` buffer is part of the output buffer and is populated
  during `llama_decode()`. Each ubatch overwrites the previous data, so the
  buffer always contains the final layer states after decoding.
- To use this feature, call `llama_set_output_layer_out(ctx, true)` **before**
  calling `llama_decode()` for the first time.
- The output file format matches the gguf-parity-tools `compare-layers` contract:
  native-endian f32 flat array of shape `[n_layers, n_embd]`.
