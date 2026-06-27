/// dump_llamacpp_layers — dump per-layer hidden states from a GGUF model via llama.cpp.
///
/// This tool uses the built-in output_layer_out API (llama_set_output_layer_out +
/// llama_save_output_layer_out), which was integrated directly into this llama.cpp
/// fork. No manual patching is required.
///
/// ## Binary output format
///
/// The output file contains concatenated per-layer hidden-state vectors for the
/// last prompt token, written in native-endian f32:
///
///   dtype:      f32 (native byte order)
///   shape:      [n_layers * n_embd]  (flat, row-major)
///   layer count: model n_layers
///   hidden size: model n_embd
///   row order:   layer 0 first, layer (n_layers-1) last
///
/// Each layer's vector is `n_embd` consecutive f32 values, taken from the last
/// token position in the sequence. The tensor boundary is the per-layer block
/// output after the final residual add and control-vector application (`l_out`
/// callback point in the model graph).
///
/// ## Build
///
///   cd /path/to/llama.cpp
///   cmake -B build -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF
///   cmake --build build --target llama -j$(nproc)
///   g++ -std=c++17 -I./include -I./ggml/include -I./src \
///       tools/dump_llamacpp_layers.cpp \
///       ./build/src/libllama.a \
///       ./build/ggml/src/libggml.a \
///       ./build/ggml/src/libggml-base.a \
///       ./build/ggml/src/libggml-cpu.a \
///       -lpthread -ldl -lm -o dump_llamacpp_layers
///
/// ## Usage
///
///   ./dump_llamacpp_layers <model.gguf> <prompt> <out.bin> [ctx_size]
///
/// Arguments:
///   model.gguf   path to GGUF model
///   prompt       text prompt (use "" for BOS-only)
///   out.bin      path for binary output
///   ctx_size     context size (default: 16)

#include "llama.h"
#include "llama-ext.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <prompt> <out.bin> [ctx_size]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * prompt     = argv[2];
    const char * out_path   = argv[3];
    int          ctx_size   = 16;

    if (argc >= 5) {
        ctx_size = atoi(argv[4]);
    }

    // --- backend init ---
    llama_backend_init();

    // --- load model ---
    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        fprintf(stderr, "error: failed to load model %s\n", model_path);
        llama_backend_free();
        return 1;
    }

    // --- create context ---
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = ctx_size;
    cp.n_seq_max = 1;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "error: failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // --- enable layer output ---
    llama_set_output_layer_out(ctx, true);

    // --- tokenize ---
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_tokens = 0;
    llama_token toks[ctx_size];
    if (strlen(prompt) == 0) {
        int bos = llama_vocab_bos(vocab);
        toks[0]  = bos;
        n_tokens = 1;
    } else {
        n_tokens = llama_tokenize(vocab, prompt, (int)strlen(prompt), toks, ctx_size, true, true);
        if (n_tokens < 0) {
            fprintf(stderr, "error: tokenize failed\n");
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
    }

    // --- decode ---
    llama_batch batch = llama_batch_get_one(toks, n_tokens);
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "error: decode failed\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // --- save layer output ---
    if (!llama_save_output_layer_out(ctx, out_path)) {
        fprintf(stderr, "error: failed to save layer output to %s\n", out_path);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    fprintf(stderr, "info: per-layer states written to %s\n", out_path);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
