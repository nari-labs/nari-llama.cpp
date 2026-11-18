#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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

// loads meta (with its tensors' data) on the CPU; the log of the load is in messages
static llama_model * load(gguf_context * meta) {
    auto * file = std::tmpfile();
    check(file != nullptr, "tmpfile failed");
    check(gguf_write_to_file_ptr(meta, file, false), "write GGUF failed");
    std::rewind(file);
    messages.clear();
    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    auto * model = llama_model_load_from_file_ptr(file, params);
    std::fclose(file);
    return model;
}

static void check_versions() {
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
        auto * model = load(meta);
        check(model == nullptr, "incomplete fixture must not load as a model");
        const bool rejected_version = messages.find("NariQ: coded tensors require trellis.version") != std::string::npos;
        check(rejected_version == (version != 1), "unexpected format version validation result");
        gguf_free(meta);
        ggml_free(ctx);
    }
}

// a one-layer llama (n_embd 256, n_vocab 32) with a coded output.weight, or a coded token_embd.weight that is also
// the output (tied); moe: coded expert weights
struct fixture {
    static constexpr int64_t n_embd = 256, n_vocab = 32, n_expert = 2;

    ggml_context * ctx;
    gguf_context * meta;

    fixture(bool tied, bool moe, ggml_type pin_type = GGML_TYPE_I32) {
        ctx = ggml_init({64 * 1024 * 1024, nullptr, false});
        meta = gguf_init_empty();
        gguf_set_val_str(meta, "general.architecture", "llama");
        gguf_set_val_u32(meta, "llama.context_length", 64);
        gguf_set_val_u32(meta, "llama.embedding_length", n_embd);
        gguf_set_val_u32(meta, "llama.block_count", 1);
        gguf_set_val_u32(meta, "llama.feed_forward_length", n_embd);
        gguf_set_val_u32(meta, "llama.attention.head_count", 2);
        gguf_set_val_u32(meta, "llama.attention.head_count_kv", 2);
        gguf_set_val_f32(meta, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
        gguf_set_val_u32(meta, "llama.vocab_size", n_vocab);
        gguf_set_val_str(meta, "tokenizer.ggml.model", "none");
        gguf_set_val_u32(meta, "trellis.version", 1);

        if (tied) {
            coded("token_embd", n_embd, n_vocab, GGML_TYPE_COUNT);
        } else {
            add("token_embd.weight", GGML_TYPE_F32, {n_embd, n_vocab});
            coded("output", n_embd, n_vocab, pin_type);
        }
        add("output_norm.weight", GGML_TYPE_F32, {n_embd});
        add("blk.0.attn_norm.weight", GGML_TYPE_F32, {n_embd});
        for (const char * name : {"attn_q", "attn_k", "attn_v", "attn_output"}) {
            add(std::string("blk.0.") + name + ".weight", GGML_TYPE_F32, {n_embd, n_embd});
        }
        add("blk.0.ffn_norm.weight", GGML_TYPE_F32, {n_embd});
        if (moe) {
            gguf_set_val_u32(meta, "llama.expert_count", n_expert);
            gguf_set_val_u32(meta, "llama.expert_used_count", 1);
            add("blk.0.ffn_gate_inp.weight", GGML_TYPE_F32, {n_embd, n_expert});
            for (const char * name : {"ffn_gate_exps", "ffn_down_exps", "ffn_up_exps"}) {
                add(std::string("blk.0.") + name + ".weight", GGML_TYPE_TRL2, {n_embd, n_embd, n_expert});
            }
        } else {
            for (const char * name : {"ffn_gate", "ffn_down", "ffn_up"}) {
                add(std::string("blk.0.") + name + ".weight", GGML_TYPE_F32, {n_embd, n_embd});
            }
        }
    }

    ~fixture() {
        gguf_free(meta);
        ggml_free(ctx);
    }

    ggml_tensor * add(const std::string & name, ggml_type type, const std::vector<int64_t> & ne) {
        auto * t = ggml_new_tensor(ctx, type, (int) ne.size(), ne.data());
        ggml_set_name(t, name.c_str());
        std::memset(t->data, 0, ggml_nbytes(t));
        if (type == GGML_TYPE_F32 && ne.size() == 1) {
            for (int64_t i = 0; i < ne[0]; ++i) {
                ((float *) t->data)[i] = 1.0f;
            }
        }
        gguf_add_tensor(meta, t);
        return t;
    }

    // a coded weight base.weight [k, n] with its sides and no rotations; pin_type GGML_TYPE_COUNT: no input gather
    void coded(const std::string & base, int64_t k, int64_t n, ggml_type pin_type) {
        add(base + ".weight", GGML_TYPE_TRL2, {k, n});
        add(base + ".trellis_s16", GGML_TYPE_F16, {k / 16});
        add(base + ".trellis_pre", GGML_TYPE_F32, {k});
        add(base + ".trellis_post", GGML_TYPE_F32, {n});
        if (pin_type == GGML_TYPE_I32) {
            auto * pin = add(base + ".trellis_pin", GGML_TYPE_I32, {k});
            for (int64_t i = 0; i < k; ++i) {
                ((int32_t *) pin->data)[i] = (int32_t) (k - 1 - i);
            }
        } else if (pin_type != GGML_TYPE_COUNT) {
            add(base + ".trellis_pin", pin_type, {k});
        }
        const int32_t desc[] = {0, 0, 1, 0, 0, 1, 1, (int32_t) n};
        gguf_set_arr_data(meta, ("trellis.desc." + base).c_str(), GGUF_TYPE_INT32, desc, 8);
    }

    int32_t * pin() {
        return (int32_t *) ggml_get_tensor(ctx, "output.trellis_pin")->data;
    }
};

// loads f; expected: a part of the rejection's message, or nullptr if the model must load
static void check_load(const char * what, fixture & f, const char * expected) {
    auto * model = load(f.meta);
    if (expected == nullptr) {
        check(model != nullptr, what);
        llama_model_free(model);
        return;
    }
    check(model == nullptr && messages.find(expected) != std::string::npos, what);
}

int main() {
    ggml_backend_load_all();
    llama_log_set(log_message, nullptr);

    check_versions();

    {
        fixture f(false, false);
        check_load("a valid coded fixture must load", f, nullptr);
    }
    {
        fixture f(false, false);
        f.pin()[3] = (int32_t) fixture::n_embd;
        check_load("an out-of-range input gather must be rejected", f, "output.trellis_pin is not a permutation");
    }
    {
        fixture f(false, false);
        f.pin()[0] = -1;
        check_load("a negative input gather must be rejected", f, "output.trellis_pin is not a permutation");
    }
    {
        fixture f(false, false);
        f.pin()[1] = f.pin()[0];
        check_load("a repeated input gather must be rejected", f, "output.trellis_pin is not a permutation");
    }
    {
        fixture f(false, false, GGML_TYPE_F32);
        check_load("a gather of the wrong type must be rejected", f, "output.trellis_pin is f32, expected i32");
    }
    {
        fixture f(true, false);
        check_load("a coded tied embedding must be rejected", f, "token_embd.weight is coded and also used as another tensor");
    }
    {
        fixture f(false, true);
        check_load("coded expert weights must be rejected", f,
                "blk.0.ffn_gate_exps.weight is coded, but only linear weights and the token embedding can be");
    }

    std::puts("trellis loading: format versions, gathers, tied embeddings and expert weights checked");
}
