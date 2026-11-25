#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

static bool verbose = false;

static void check_close(float actual, double expected) {
    check(std::isfinite(actual) && std::abs(actual - expected) < 2e-4 * (1 + std::abs(expected)), "trellis numerical mismatch");
}

static void run(ggml_backend_t backend, int bits, int mode, int columns) {
    constexpr int K = 256, N = 128;
    auto * ctx = ggml_init({16 * 1024 * 1024, nullptr, false});
    check(ctx != nullptr, "context allocation failed");
    auto type = static_cast<ggml_type>(GGML_TYPE_TRL1 + bits - 1);
    auto * w = ggml_new_tensor_2d(ctx, type, K, N);
    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, columns);
    auto * s = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, K / 16);
    auto * pre = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K);
    auto * post = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    auto * pin = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, K);
    auto * pout = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    std::mt19937 rng(1234 + bits);
    for (size_t i = 0; i < ggml_nbytes(w); ++i) { static_cast<unsigned char *>(w->data)[i] = rng() & 255; }
    for (int i = 0; i < K * columns; ++i) { static_cast<float *>(x->data)[i] = float(int(rng() % 101) - 50) / 100; }
    for (int i = 0; i < K / 16; ++i) { static_cast<ggml_fp16_t *>(s->data)[i] = ggml_fp32_to_fp16(0.1f + (i % 7) * 0.01f); }
    for (int i = 0; i < K; ++i) {
        static_cast<float *>(pre->data)[i] = (i % 2 ? -1.f : 1.f) * (1.f + (i % 3) * 0.1f);
        static_cast<int32_t *>(pin->data)[i] = K - 1 - i;
    }
    for (int i = 0; i < N; ++i) {
        static_cast<float *>(post->data)[i] = 0.8f + (i % 5) * 0.1f;
        static_cast<int32_t *>(pout->data)[i] = N - 1 - i;
    }
    int32_t desc[] = {mode ? 1 : 0, mode ? 128 : 0, 1, mode ? 1 : 0, mode ? 128 : 0, 1, 1, N};
    auto * ip = mode == 2 ? pin : nullptr;
    auto * op = mode == 2 ? pout : nullptr;
    auto * y = ggml_trellis_mul_mat(ctx, w, x, s, pre, post, ip, op, desc, 8);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "trellis graph failed");
    std::vector<float> dense(K * N);
    check(ggml_trellis_dequantize(type, w->data, K, N, static_cast<ggml_fp16_t *>(s->data),
        static_cast<float *>(pre->data), static_cast<float *>(post->data),
        ip ? static_cast<int32_t *>(ip->data) : nullptr, op ? static_cast<int32_t *>(op->data) : nullptr,
        desc, 8, dense.data()), "dequantize failed");
    for (int c = 0; c < columns; ++c) {
        for (int n = 0; n < N; ++n) {
            double expected = 0;
            for (int k = 0; k < K; ++k) { expected += double(dense[n * K + k]) * static_cast<float *>(x->data)[c * K + k]; }
            check_close(static_cast<float *>(y->data)[c * N + n], expected);
        }
    }

    // Embeddings rotate columns only. Include both ordinary and frequency-banded lookups.
    desc[0] = desc[1] = 0;
    check(ggml_trellis_dequantize(type, w->data, K, N, static_cast<ggml_fp16_t *>(s->data),
        static_cast<float *>(pre->data), static_cast<float *>(post->data), nullptr, nullptr,
        desc, 8, dense.data()), "embedding decode failed");
    auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    const int32_t rows[] = {0, 17, N - 1, 32};
    for (int i = 0; i < 4; ++i) { static_cast<int32_t *>(ids->data)[i] = rows[i]; }
    auto * e = ggml_trellis_get_rows(ctx, w, ids, s, pre, post, desc, 8);
    graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, e);
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "embedding graph failed");
    for (int i = 0; i < 4; ++i) {
        for (int k = 0; k < K; ++k) { check_close(static_cast<float *>(e->data)[i * K + k], dense[rows[i] * K + k]); }
    }
    auto * gather = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2 * N);
    auto * post2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    for (int i = 0; i < N; ++i) { static_cast<float *>(post2->data)[i] = -static_cast<float *>(post->data)[i]; }
    for (int i = 0; i < 2 * N; ++i) { static_cast<int32_t *>(gather->data)[i] = 2 * N - 1 - i; }
    const int32_t tokens[] = {0, 17, N + 17, 2 * N - 1};
    for (int i = 0; i < 4; ++i) { static_cast<int32_t *>(ids->data)[i] = tokens[i]; }
    auto * band = ggml_trellis_get_rows_band(ctx, nullptr, w, ids, gather, 0, s, pre, post, desc, 8);
    band = ggml_trellis_get_rows_band(ctx, band, w, ids, gather, N, s, pre, post2, desc, 8);
    graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, band);
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "banded embedding graph failed");
    for (int i = 0; i < 4; ++i) {
        int row = 2 * N - 1 - tokens[i];
        for (int k = 0; k < K; ++k) { check_close(static_cast<float *>(band->data)[i * K + k], (row >= N ? -1 : 1) * dense[(row % N) * K + k]); }
    }
    ggml_free(ctx);
}


// Device tests use backend allocations and transfers, never tensor->data for device results.
static void run_device(ggml_backend_t backend, int bits, int mode, int columns, bool high_range = false) {
    const int K = mode == 4 ? 192 : mode == 5 ? 1088 : mode == 6 ? 384 : mode == 8 ? 272 : 256;
    const int N = mode == 4 ? 320 : mode == 5 ? 1088 : mode == 6 ? 640 : mode == 8 ? 304 : mode == 9 ? 48 : 128;
    if (verbose) {
        std::printf("%s: TRL%d mode=%d K=%d N=%d tokens=%d\n", ggml_backend_name(backend), bits, mode, K, N, columns);
    }
    auto * ctx = ggml_init({16*1024*1024, nullptr, true});
    check(ctx != nullptr, "device context allocation failed");
    const auto type = static_cast<ggml_type>(GGML_TYPE_TRL1 + bits - 1);
    auto * w = ggml_new_tensor_2d(ctx, type, K, N);
    // Exercise non-contiguous rows and higher-dimensional batches as well.
    auto * storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K + 16, columns);
    auto * x = ggml_view_2d(ctx, storage, K, columns, (K + 16)*sizeof(float), 0);
    if (mode == 7) { x = ggml_view_4d(ctx, storage, K, 1, 1, columns, storage->nb[1], storage->nb[1], storage->nb[1], 0); }
    auto * scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, K/16);
    auto * pre = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K);
    auto * post = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    auto * pin = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, K);
    auto * pout = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    int32_t desc[] = {0, 0, 1, 0, 0, 1, 1, N, 0};
    if (mode == 1 || mode == 2 || mode == 7) { desc[0] = desc[3] = 1; desc[1] = desc[4] = 128; }
    if (mode >= 3 && mode <= 6) { desc[0] = desc[3] = 2; }
    if (mode == 6) { desc[2] = desc[5] = 2; desc[6] = 2; desc[7] = 256; desc[8] = 384; }
    if (mode == 10) { desc[0] = desc[3] = 2; desc[6] = 0; }
    const int nd = 7 + desc[6];
    const bool permute = mode >= 2;
    auto * y = ggml_trellis_mul_mat(ctx, w, x, scale, pre, post, permute ? pin : nullptr, permute ? pout : nullptr, desc, nd);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    check(ggml_backend_supports_op(backend, y), "backend rejected valid coded product");
    if (std::string(ggml_backend_name(backend)).find("MTL") == 0) {
        const size_t stride = x->nb[0];
        x->nb[0] = 2*sizeof(float);
        check(!ggml_backend_supports_op(backend, y), "Metal accepted strided elements");
        x->nb[0] = stride;
        const int32_t kind = y->op_params[0];
        y->op_params[0] = 9;
        check(!ggml_backend_supports_op(backend, y), "Metal accepted unsupported transform");
        y->op_params[0] = kind;
    }
    if (bits == 1 && mode == 0 && columns == 1 && std::string(ggml_backend_name(backend)).find("ROCm") == 0) {
        const size_t stride = w->nb[1];
        w->nb[1] += w->nb[0];
        check(!ggml_backend_supports_op(backend, y), "ROCm accepted a strided coded weight");
        w->nb[1] = stride;
        w->ne[1] = 0;
        check(!ggml_backend_supports_op(backend, y), "ROCm accepted an empty coded weight");
        w->ne[1] = N;
    }
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(buffer != nullptr, "device allocation failed");
    std::mt19937 rng(1234 + bits);
    std::vector<unsigned char> codes(ggml_nbytes(w));
    std::vector<float> input((K + 16)*columns), a(K), b(N);
    std::vector<int32_t> ip(K), op(N);
    std::vector<ggml_fp16_t> scales(K/16);
    for (auto & v : codes) { v = rng() & 255; }
    for (auto & v : input) { v = float(int(rng()%101) - 50)/100; }
    if (high_range) {
        for (int c : {0, columns - 1}) {
            float * row = input.data() + c * (K + 16);
            std::fill_n(row, K, 0.0f);
            row[0] = 1000000.0f;
        }
    }
    for (int i = 0; i < K/16; ++i) { scales[i] = ggml_fp32_to_fp16(0.1f + (i%7)*0.01f); }
    for (int i = 0; i < K; ++i) { a[i] = (i%2 ? -1.f : 1.f)*(1.f + (i%3)*0.1f); ip[i] = K - 1 - i; }
    for (int i = 0; i < N; ++i) { b[i] = 0.8f + (i%5)*0.1f; op[i] = N - 1 - i; }
    ggml_backend_tensor_set(w, codes.data(), 0, codes.size());
    ggml_backend_tensor_set(storage, input.data(), 0, input.size()*sizeof(float));
    ggml_backend_tensor_set(scale, scales.data(), 0, scales.size()*sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(pre, a.data(), 0, a.size()*sizeof(float));
    ggml_backend_tensor_set(post, b.data(), 0, b.size()*sizeof(float));
    ggml_backend_tensor_set(pin, ip.data(), 0, ip.size()*sizeof(int32_t));
    ggml_backend_tensor_set(pout, op.data(), 0, op.size()*sizeof(int32_t));
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "device graph failed");
    std::vector<float> result(N*columns), dense(K*N);
    ggml_backend_tensor_get(y, result.data(), 0, result.size()*sizeof(float));
    check(ggml_trellis_dequantize(type, codes.data(), K, N, scales.data(), a.data(), b.data(),
        permute ? ip.data() : nullptr, permute ? op.data() : nullptr, desc, nd, dense.data()), "reference decode failed");
    for (int c = 0; c < columns; ++c) {
        for (int n = 0; n < N; ++n) {
            double expected = 0;
            for (int k = 0; k < K; ++k) { expected += double(dense[n*K + k])*input[c*(K + 16) + k]; }
            if (!std::isfinite(result[c*N + n]) || std::abs(result[c*N + n] - expected) >= 2e-4*(1 + std::abs(expected))) {
                std::fprintf(stderr, "TRL%d mode=%d token=%d row=%d got=%g expected=%g\n", bits, mode, c, n, result[c*N + n], expected);
                check(false, "device numerical mismatch");
            }
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static std::vector<float> run_mlp_chain(ggml_backend_t backend, int columns) {
    constexpr int width = 256;
    auto * ctx = ggml_init({1024 * 1024, nullptr, true});
    check(ctx != nullptr, "MLP context allocation failed");
    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, columns);
    auto * s16 = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, width / 16);
    auto * pre = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
    auto * post = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
    auto * gate_w = ggml_new_tensor_2d(ctx, GGML_TYPE_TRL2, width, width);
    auto * up_w = ggml_new_tensor_2d(ctx, GGML_TYPE_TRL2, width, width);
    auto * down_w = ggml_new_tensor_2d(ctx, GGML_TYPE_TRL2, width, width);
    const int32_t desc[] = {1, 128, 1, 1, 128, 1, 0};
    auto * gate = ggml_trellis_mul_mat(ctx, gate_w, x, s16, pre, post, nullptr, nullptr, desc, 7);
    auto * up = ggml_trellis_mul_mat(ctx, up_w, x, s16, pre, post, nullptr, nullptr, desc, 7);
    auto * glu = ggml_swiglu_split(ctx, gate, up);
    auto * down = ggml_trellis_mul_mat(ctx, down_w, glu, s16, pre, post, nullptr, nullptr, desc, 7);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, down);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(buffer != nullptr, "MLP buffer allocation failed");

    std::mt19937 rng(3251);
    std::vector<unsigned char> codes(ggml_nbytes(gate_w));
    for (auto * w : {gate_w, up_w, down_w}) {
        for (auto & v : codes) {
            v = rng() & 255;
        }
        ggml_backend_tensor_set(w, codes.data(), 0, codes.size());
    }
    const std::vector<float> side(width, 0.75f);
    const std::vector<ggml_fp16_t> scales(width / 16, ggml_fp32_to_fp16(0.005f));
    std::vector<float> data(width * columns);
    for (auto & v : data) {
        v = float(int(rng() % 101) - 50) / 100.0f;
    }
    ggml_backend_tensor_set(s16, scales.data(), 0, ggml_nbytes(s16));
    ggml_backend_tensor_set(pre, side.data(), 0, ggml_nbytes(pre));
    ggml_backend_tensor_set(post, side.data(), 0, ggml_nbytes(post));
    ggml_backend_tensor_set(x, data.data(), 0, ggml_nbytes(x));
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "MLP graph failed");
    ggml_backend_tensor_get(down, data.data(), 0, ggml_nbytes(down));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return data;
}


static std::vector<float> run_hybrid(ggml_backend_t device, ggml_backend_t cpu, int columns, bool bands) {
    constexpr int K = 256, N = 128;
    auto * host = ggml_init({1024*1024, nullptr, true});
    auto * weights = ggml_init({1024*1024, nullptr, true});
    auto * ctx = ggml_init({4*1024*1024, nullptr, true});
    check(host && weights && ctx, "hybrid context allocation failed");
    auto weight = [](ggml_context * c, int bits) {
        return std::vector<ggml_tensor *>{
            ggml_new_tensor_2d(c, static_cast<ggml_type>(GGML_TYPE_TRL1 + bits - 1), K, N),
            ggml_new_tensor_1d(c, GGML_TYPE_F16, K/16),
            ggml_new_tensor_1d(c, GGML_TYPE_F32, K), ggml_new_tensor_1d(c, GGML_TYPE_F32, N)};
    };
    auto emb = weight(host, 2), head = weight(weights, 3), head2 = weight(weights, 6);
    auto * ids = ggml_new_tensor_1d(host, GGML_TYPE_I32, columns);
    auto * gather = ggml_new_tensor_1d(host, GGML_TYPE_I32, 2*N);
    auto * vocab = ggml_new_tensor_1d(weights, GGML_TYPE_I32, 2*N);
    const int32_t ed[] = {0, 0, 1, 1, 128, 1, 1, N};
    ggml_tensor * e;
    if (bands) {
        e = ggml_trellis_get_rows_band(ctx, nullptr, emb[0], ids, gather, 0, emb[1], emb[2], emb[3], ed, 8);
        e = ggml_trellis_get_rows_band(ctx, e, emb[0], ids, gather, N, emb[1], emb[2], emb[3], ed, 8);
    } else {
        e = ggml_trellis_get_rows(ctx, emb[0], ids, emb[1], emb[2], emb[3], ed, 8);
    }
    if (device != cpu && ggml_backend_supports_op(device, e)) {
        std::fprintf(stderr, "%s advertises coded embeddings\n", ggml_backend_name(device));
        std::exit(1);
    }
    const int32_t d[] = {1, 128, 1, 1, 128, 1, 1, N};
    auto * a = ggml_trellis_mul_mat(ctx, head[0], e, head[1], head[2], head[3], nullptr, nullptr, d, 8);
    auto * b = ggml_trellis_mul_mat(ctx, head2[0], e, head2[1], head2[2], head2[3], nullptr, nullptr, d, 8);
    auto * y = ggml_concat(ctx, a, b, 0);
    if (columns == 1) {
        y = ggml_reshape_2d(ctx, y, 1, 2*N);
        y = ggml_get_rows(ctx, y, vocab);
        y = ggml_reshape_2d(ctx, y, 2*N, 1);
    } else {
        y = ggml_cont(ctx, ggml_transpose(ctx, y));
        y = ggml_get_rows(ctx, y, vocab);
        y = ggml_cont(ctx, ggml_transpose(ctx, y));
    }
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    auto * hbuf = ggml_backend_alloc_ctx_tensors(host, cpu);
    auto * wbuf = ggml_backend_alloc_ctx_tensors(weights, device);
    check(hbuf && wbuf, "hybrid weight allocation failed");
    ggml_backend_buffer_set_usage(hbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    int seed = 17;
    for (const auto & v : {emb, head, head2}) {
        std::mt19937 rng(seed++);
        std::vector<unsigned char> codes(ggml_nbytes(v[0]));
        for (auto & q : codes) { q = rng() & 255; }
        std::vector<ggml_fp16_t> scales(K/16, ggml_fp32_to_fp16(0.01f));
        std::vector<float> pre(K, 0.5f), post(N, 0.25f);
        ggml_backend_tensor_set(v[0], codes.data(), 0, codes.size());
        ggml_backend_tensor_set(v[1], scales.data(), 0, scales.size()*sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(v[2], pre.data(), 0, pre.size()*sizeof(float));
        ggml_backend_tensor_set(v[3], post.data(), 0, post.size()*sizeof(float));
    }
    std::vector<int32_t> tokens(columns), perm(2*N);
    for (int i = 0; i < columns; ++i) { tokens[i] = (i*37)%(bands ? 2*N : N); }
    for (int i = 0; i < 2*N; ++i) { perm[i] = 2*N - 1 - i; }
    ggml_backend_tensor_set(ids, tokens.data(), 0, tokens.size()*sizeof(int32_t));
    ggml_backend_tensor_set(gather, perm.data(), 0, perm.size()*sizeof(int32_t));
    ggml_backend_tensor_set(vocab, perm.data(), 0, perm.size()*sizeof(int32_t));
    ggml_backend_t backends[] = {device, cpu};
    auto * sched = ggml_backend_sched_new(backends, nullptr, device == cpu ? 1 : 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    check(ggml_backend_sched_alloc_graph(sched, graph), "hybrid graph allocation failed");
    check(ggml_backend_sched_get_tensor_backend(sched, e) == cpu, "embedding fallback was not assigned to CPU");
    check(ggml_backend_sched_get_tensor_backend(sched, a) == device, "coded head did not offload");
    check(ggml_backend_sched_get_tensor_backend(sched, b) == device, "second coded head did not offload");
    check(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "hybrid graph failed");
    std::vector<float> result(2*N*columns);
    ggml_backend_tensor_get(y, result.data(), 0, result.size()*sizeof(float));
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(wbuf);
    ggml_backend_buffer_free(hbuf);
    ggml_free(ctx); ggml_free(weights); ggml_free(host);
    return result;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    auto * cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    check(cpu != nullptr, "CPU backend unavailable");
    ggml_backend_t device = cpu;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-v") {
            verbose = true;
        } else if (device == cpu) {
            device = ggml_backend_init_by_name(argv[i], nullptr);
            check(device != nullptr, "requested backend unavailable");
            if (ggml_backend_dev_type(ggml_backend_get_device(device)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                ggml_backend_free(device);
                device = cpu;
            }
        } else {
            check(false, "usage: test-trellis [-v] [backend-name]");
        }
    }
    const int32_t invalid[] = {1, 3, 1, 0, 0, 1, 1, 128};
    check(!ggml_trellis_desc_supported(256, 128, invalid, 8), "invalid Hadamard block accepted");
    check(!ggml_trellis_desc_supported(256, 128, nullptr, 0), "missing descriptor accepted");
    {
        // a coded weight has no per-row decode or dot product: only the trellis ops may read it
        auto * ctx = ggml_init({1024 * 1024, nullptr, true});
        auto * w = ggml_new_tensor_2d(ctx, GGML_TYPE_TRL2, 256, 128);
        auto * e = ggml_new_tensor_3d(ctx, GGML_TYPE_TRL2, 256, 128, 2);
        auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 4);
        auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
        auto * eids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 4);
        ggml_tensor * ops[] = {
            ggml_get_rows(ctx, w, ids),
            ggml_mul_mat(ctx, w, x),
            ggml_mul_mat_id(ctx, e, ggml_reshape_3d(ctx, x, 256, 1, 4), eids),
        };
        for (ggml_backend_t backend : {cpu, device}) {
            for (ggml_tensor * op : ops) {
                if (ggml_backend_supports_op(backend, op)) {
                    std::fprintf(stderr, "%s accepts %s on a coded weight\n", ggml_backend_name(backend), ggml_op_name(op->op));
                    std::exit(1);
                }
            }
        }
        ggml_free(ctx);
    }
    for (int bits = 1; bits <= 6; ++bits) {
        for (int mode = 0; mode < 3; ++mode) {
            for (int columns : {1, 7}) { run(cpu, bits, mode, columns); }
        }
    }
    for (int bits = 1; bits <= 6; ++bits) {
        for (int mode = 0; mode < 9; ++mode) {
            for (int columns : {1, 7, 35}) { run_device(device, bits, mode, columns); }
        }
    }
    if (std::string(ggml_backend_name(device)).find("ROCm") == 0) {
        for (int bits = 1; bits <= 6; ++bits) {
            for (int columns : {8, 31, 32}) { run_device(device, bits, 0, columns); }
        }
    }
    if (device == cpu || std::string(ggml_backend_name(device)).find("ROCm") == 0) {
        run_device(device, 3, 0, 7, true);
        run_device(device, 3, 9, 35, true);
        run_device(device, 3, 0, 65, true);
        run_device(device, 3, 10, 35);
        for (int columns : {1, 35}) {
            const auto expected = run_mlp_chain(cpu, columns);
            const auto actual = run_mlp_chain(device, columns);
            for (size_t i = 0; i < actual.size(); ++i) {
                check_close(actual[i], expected[i]);
            }
        }
    }
    for (bool bands : {false, true}) {
        for (int columns : {1, 7, 35}) {
            auto expected = run_hybrid(cpu, cpu, columns, bands);
            auto actual = run_hybrid(device, cpu, columns, bands);
            for (size_t i = 0; i < actual.size(); ++i) { check_close(actual[i], expected[i]); }
        }
    }
    if (device != cpu) { ggml_backend_free(device); }
    ggml_backend_free(cpu);
    std::puts("TRL1-6: CPU embeddings/bands and backend products/transforms/permutations/batches passed");
}
