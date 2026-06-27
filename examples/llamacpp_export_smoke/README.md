# llama.cpp Layer Export Smoke

Minimal smoke test proving the llama.cpp per-layer hidden-state dump feature
produces outputs compatible with [gguf-parity-tools](https://github.com/voidwest/gguf-parity-tools)
and the [Sarf](https://github.com/voidwest/ember) artifact manifest contract.

## Success claim

**Qwen3-0.6B and Llama-3.2-1B layer-output dumps are emitted in a
gguf-parity-tools-compatible format and can be wrapped as Sarf artifact
manifests.**

## Verified models

| Model                       | Layers | Embed dim | Dump size       | Verified |
|-----------------------------|--------|-----------|-----------------|----------|
| Qwen3-0.6B (Q8_0)           | 28     | 1024      | 114,688 bytes   | Yes      |
| Llama-3.2-1B-Instruct (Q8_0)| 16     | 2048      | 131,072 bytes   | Yes      |

## Quick start

```sh
bash examples/llamacpp_export_smoke/commands.sh
```

## Format contract

The llama.cpp layer dump is a flat native-endian `float32` binary:

- **shape**: `[n_layers * n_embd]` flat, row-major
- **layer order**: layer 0 first, layer (n_layers - 1) last
- **capture point**: block output after residual add and control-vector
  application (`l_out` callback in the model graph)
- **token position**: last prompt token

This binary can be directly consumed by:

1. `gguf-parity tools compare-layers` (reference dump)
2. A Sarf manifest wrapper that shards the flat dump into per-layer `.npy`
   files for the `ember.layer_sharded_npy.v1` layout

## Files in this smoke

| File                    | Purpose                                    |
|-------------------------|--------------------------------------------|
| `README.md`             | This file                                  |
| `commands.sh`           | Exact reproduction commands                |
| `sample_manifest.json`  | Sarf artifact manifest wrapping the dump   |
| `expected_shapes.md`    | Per-model tensor shape reference           |
