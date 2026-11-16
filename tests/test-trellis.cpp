#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
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

static void close(float actual, double expected) {
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
            close(static_cast<float *>(y->data)[c * N + n], expected);
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
        for (int k = 0; k < K; ++k) { close(static_cast<float *>(e->data)[i * K + k], dense[rows[i] * K + k]); }
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
        for (int k = 0; k < K; ++k) { close(static_cast<float *>(band->data)[i * K + k], (row >= N ? -1 : 1) * dense[(row % N) * K + k]); }
    }
    ggml_free(ctx);
}

int main() {
    ggml_backend_load_all();
    auto * backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    check(backend != nullptr, "CPU backend unavailable");
    const int32_t invalid[] = {1, 3, 1, 0, 0, 1, 1, 128};
    check(!ggml_trellis_desc_supported(256, 128, invalid, 8), "invalid Hadamard block accepted");
    check(!ggml_trellis_desc_supported(256, 128, nullptr, 0), "missing descriptor accepted");
    for (int bits = 1; bits <= 6; ++bits) {
        for (int mode = 0; mode < 3; ++mode) {
            for (int columns : {1, 7}) { run(backend, bits, mode, columns); }
        }
    }
    ggml_backend_free(backend);
    std::puts("TRL1-6: products, rotations, permutations, embedding rows and bands passed");
}
