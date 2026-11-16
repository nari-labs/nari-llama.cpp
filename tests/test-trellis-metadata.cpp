#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <string>

static std::string messages;

static void log_message(ggml_log_level, const char * text, void *) {
    messages += text;
}

static void check(bool ok, const char * message) {
    if (!ok) {
        std::fprintf(stderr, "%s\n%s", message, messages.c_str());
        std::exit(1);
    }
}

int main() {
    ggml_backend_load_all();
    llama_log_set(log_message, nullptr);
    for (int version : {-1, 0, 1, 2, 3}) {
        auto * ctx = ggml_init({1024 * 1024, nullptr, false});
        auto * w = ggml_new_tensor_2d(ctx, GGML_TYPE_TRL1, 256, 16);
        ggml_set_name(w, "token_embd.weight");
        auto * meta = gguf_init_empty();
        gguf_set_val_str(meta, "general.architecture", "llama");
        if (version == 3) {
            gguf_set_val_str(meta, "trellis.version", "1");
        } else if (version >= 0) {
            gguf_set_val_u32(meta, "trellis.version", version);
        }
        gguf_add_tensor(meta, w);
        auto * file = std::tmpfile();
        check(file != nullptr, "tmpfile failed");
        check(gguf_write_to_file_ptr(meta, file, false), "write GGUF failed");
        std::rewind(file);
        messages.clear();
        auto params = llama_model_default_params();
        params.n_gpu_layers = 0;
        params.load_mode = LLAMA_LOAD_MODE_NONE;
        auto * model = llama_model_load_from_file_ptr(file, params);
        check(model == nullptr, "incomplete fixture must not load as a model");
        const bool rejected_version = messages.find("NariQ: coded tensors require trellis.version") != std::string::npos;
        check(rejected_version == (version != 1), "unexpected format version validation result");
        std::fclose(file);
        gguf_free(meta);
        ggml_free(ctx);
    }
    std::puts("NariQ format metadata: missing, unsupported and wrong-type versions rejected");
}
