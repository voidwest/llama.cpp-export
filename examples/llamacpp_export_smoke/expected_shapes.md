# Expected Tensor Shapes

llama.cpp per-layer hidden-state dumps for verified model + prompt pairs.
All dumps are native-endian float32 flat binaries.

## Qwen3-0.6B (qwen3 architecture)

| Parameter    | Value        |
|--------------|--------------|
| n_layers     | 28           |
| n_embd       | 1024         |
| total floats | 28,672       |
| total bytes  | 114,688      |

### Per-layer shapes (Sarf sharded)

Each layer is a single sample (one prompt, last token):

```
layers/layer_0000.npy  shape: [1, 1024]  dtype: float32
layers/layer_0001.npy  shape: [1, 1024]  dtype: float32
...
layers/layer_0027.npy  shape: [1, 1024]  dtype: float32
```

### Flat dump layout

```
Offset 0:            layer_0[0] ... layer_0[1023]   (1024 floats)
Offset 1024:         layer_1[0] ... layer_1[1023]   (1024 floats)
...
Offset 27*1024:      layer_27[0] ... layer_27[1023] (1024 floats)
```

## Llama-3.2-1B (llama architecture)

| Parameter    | Value        |
|--------------|--------------|
| n_layers     | 16           |
| n_embd       | 2048         |
| total floats | 32,768       |
| total bytes  | 131,072      |

### Per-layer shapes (Sarf sharded)

```
layers/layer_0000.npy  shape: [1, 2048]  dtype: float32
layers/layer_0001.npy  shape: [1, 2048]  dtype: float32
...
layers/layer_0015.npy  shape: [1, 2048]  dtype: float32
```

### Flat dump layout

```
Offset 0:            layer_0[0] ... layer_0[2047]   (2048 floats)
Offset 2048:         layer_1[0] ... layer_1[2047]   (2048 floats)
...
Offset 15*2048:      layer_15[0] ... layer_15[2047] (2048 floats)
```

## Capture point

For both architectures, the captured tensor is `cur` after `build_cvec(cur, il)`
in the model graph — the block output after the residual add and control-vector
application. This is the `cb(cur, "l_out", il)` callback point.

## Format verification

```sh
# Verify flat dump shape
python3 -c "
import numpy as np
d = np.fromfile('layers.bin', dtype=np.float32)
print(f'n_float={len(d)}  expected={n_layers * n_embd}')
assert len(d) == n_layers * n_embd
"

# Verify with gguf-parity-tools
gguf-parity compare-layers \
    --candidate layers.bin \
    --reference layers.bin \
    --layers 28 --hidden-size 1024 \
    --out report
# Expected: status=pass, shape_check.matches=true
```
