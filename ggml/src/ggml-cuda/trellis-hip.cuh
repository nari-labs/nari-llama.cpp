#include <cmath>

// Reference HIP implementation of the NariQ coded linear.
namespace {
constexpr int threads = 128;
constexpr int rows_per_block = threads / WARP_SIZE;
constexpr int gemm_min_tokens = 32;
constexpr int gemm_stripe_rows = 4096;

struct side_layout {
    int width;
    int slice;
    int parts;
    int starts[GGML_TRELLIS_MAX_PARTS];
    int sizes[GGML_TRELLIS_MAX_PARTS];

    __device__ int stored(int index) const {
        const int shard = index / slice;
        int local = index % slice;
        for (int i = 0; i < parts - 1; ++i) {
            if (local < sizes[i]) return starts[i] + shard * sizes[i] + local;
            local -= sizes[i];
        }
        return starts[parts - 1] + shard * sizes[parts - 1] + local;
    }
};

static __global__ void scale_gather(const float * src, float * dst, const float * scale,
                                    const int32_t * gather, int width, size_t stride, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const int j = i % width;
        const int k = gather ? gather[j] : j;
        dst[i] = src[i / width * stride + k] * (scale ? scale[k] : 1.0f);
    }
}

static __global__ void pack_side(const float * src, float * dst, side_layout layout, bool inverse, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const size_t stored = i - i % layout.width + layout.stored(i % layout.width);
        if (inverse) dst[stored] = src[i];
        else dst[i] = src[stored];
    }
}

static __global__ void butterfly(float * data, int step, size_t pairs) {
    const size_t p = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p < pairs) {
        const size_t i = p / step * (2 * step) + p % step;
        const float a = data[i], b = data[i + step];
        data[i] = a + b;
        data[i + step] = a - b;
    }
}

static __global__ void normalize(float * data, float factor, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) data[i] *= factor;
}

struct paley_signs {
    uint64_t low[68]{}, high[68]{};
};

static __global__ void paley(const float * src, float * dst, paley_signs signs,
                             int m, int run, float factor, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int a = i / run % m;
    const size_t base = i - a * run;
    float sum = 0;
    for (int b = 0; b < m; ++b) {
        const bool positive = b < 64 ? (signs.low[a] >> b) & 1 : (signs.high[a] >> (b - 64)) & 1;
        const float value = src[base + b * run];
        sum += positive ? value : -value;
    }
    dst[i] = sum * factor;
}

static __global__ void fold_scales(float * data, const half * scale, int width, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) data[i] *= __half2float(scale[(i % width) / 16]);
}

template<int bits>
static __device__ int decode_value(const uint32_t * tile, int r, int c) {
    const int p = 8 * (4 * (r % 8) + (c % 8) / 2) + 4 * (r / 8) + 2 * (c / 8) + c % 2;
    const int start = ((p + 1) * bits - 16 + 256 * bits) % (256 * bits);
    const uint64_t pair = (uint64_t(tile[start / 32]) << 32) | tile[(start / 32 + 1) % (8 * bits)];
    const uint32_t v = uint32_t((pair >> (48 - start % 32)) & 65535u) * 0x83DCD12Du;
    return int(((v >> 2) & 63) + ((v >> 10) & 63) + ((v >> 18) & 63) + ((v >> 26) & 63)) - 126;
}

// Four independent 32-lane row reductions per workgroup.
template<int bits>
static __global__ void coded_product(const uint32_t * codes, const float * x, float * y,
                                     int k, int n, size_t count) {
    constexpr int lanes = WARP_SIZE;
    const int lane = threadIdx.x % lanes;
    const size_t index = size_t(blockIdx.x) * rows_per_block + threadIdx.x / lanes;
    if (index >= count) return;
    const int row = index % n;
    const int r = row % 16;
    x += index / n * k;
    float sum = 0;
    for (int column = lane; column < k; column += lanes) {
        const auto * tile = codes + (size_t(row / 16) * (k / 16) + column / 16) * (8 * bits);
        sum += decode_value<bits>(tile, r, column % 16) * x[column];
    }
    for (int delta = lanes / 2; delta > 0; delta /= 2) {
        sum += __shfl_down(sum, delta, lanes);
    }
    if (lane == 0) y[index] = sum;
}

// Decode one 16x16 weight tile into LDS and share it across 16 tokens.
template<int bits>
static __global__ void coded_tile_product(const uint32_t * codes, const float * x, float * y,
                                          int k, int n, size_t tokens) {
    const int row = threadIdx.x;
    const int token = threadIdx.y;
    const size_t row_tile = blockIdx.x % (n / 16);
    const size_t t = blockIdx.x / (n / 16) * 16 + token;
    __shared__ float weights[16][17];
    __shared__ float input[16][17];
    float sum = 0;
    for (int column = 0; column < k; column += 16) {
        const auto * tile = codes + (row_tile * (k / 16) + column / 16) * (8 * bits);
        weights[row][token] = decode_value<bits>(tile, row, token);
        input[token][row] = t < tokens ? x[t * k + column + row] : 0.0f;
        __syncthreads();
        for (int i = 0; i < 16; ++i) sum += weights[row][i] * input[token][i];
        __syncthreads();
    }
    if (t < tokens) y[t * n + row_tile * 16 + row] = sum;
}

template<int bits>
static void launch_product(const uint32_t * codes, const float * x, float * y,
                           int k, int n, size_t tokens, cudaStream_t stream) {
    if (tokens >= 8) {
        coded_tile_product<bits><<<((tokens + 15) / 16) * (n / 16), dim3(16, 16), 0, stream>>>(
            codes, x, y, k, n, tokens);
    } else {
        const size_t count = tokens * n;
        coded_product<bits><<<(count + rows_per_block - 1) / rows_per_block, threads, 0, stream>>>(
            codes, x, y, k, n, count);
    }
}

static unsigned blocks(size_t count) {
    return (count + threads - 1) / threads;
}

template<int bits>
static __global__ void unpack_weights(const uint32_t * codes, float * out, int k, int first, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int row = first + i / k;
    const int col = i % k;
    const auto * tile = codes + (size_t(row / 16) * (k / 16) + col / 16) * (8 * bits);
    out[i] = float(decode_value<bits>(tile, row % 16, col % 16));
}

template<int bits>
static void product(ggml_backend_cuda_context & ctx, const uint32_t * codes, const float * x,
                    float * y, int k, int n, size_t tokens) {
    const auto stream = ctx.stream();
    if (tokens < gemm_min_tokens) {
        launch_product<bits>(codes, x, y, k, n, tokens, stream);
        return;
    }
    // Bound temporary memory for large output-vocabulary bands.
    const int stripe = std::min(n, gemm_stripe_rows);
    // Keep the transformed activations in FP32, including near cancellation.
    ggml_cuda_pool_alloc<float> weights(ctx.pool(), size_t(stripe) * k);
    const float alpha = 1.0f, beta = 0.0f;
    auto handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    for (int row = 0; row < n; row += stripe) {
        const int count = std::min(stripe, n - row);
        const size_t elements = size_t(count) * k;
        unpack_weights<bits><<<blocks(elements), threads, 0, stream>>>(codes, weights.ptr, k, row, elements);
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            count, tokens, k, &alpha, weights.ptr, CUDA_R_32F, k, x, CUDA_R_32F, k,
            &beta, y + row, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
}

static void fwht(float * data, size_t count, int run, cudaStream_t stream) {
    for (int step = 1; step < run; step *= 2) {
        butterfly<<<blocks(count / 2), threads, 0, stream>>>(data, step, count / 2);
    }
    if (run > 1) normalize<<<blocks(count), threads, 0, stream>>>(data, float(1.0 / std::sqrt(double(run))), count);
}

static paley_signs make_signs(int m, bool transpose) {
    paley_signs result;
    const int q = m - 1;
    bool square[68]{};
    for (int i = 1; i < q; ++i) square[i * i % q] = true;
    for (int a = 0; a < m; ++a) {
        for (int b = 0; b < m; ++b) {
            const int i = transpose ? b : a;
            const int j = transpose ? a : b;
            if (i == j || i == 0 || (j != 0 && square[(j - i + q) % q])) {
                if (b < 64) result.low[a] |= uint64_t(1) << b;
                else result.high[a] |= uint64_t(1) << (b - 64);
            }
        }
    }
    return result;
}

static void apply_side(ggml_backend_cuda_context & ctx, float * data, int width, size_t count,
                       const int32_t * side, const int32_t * parts, int nparts, bool inverse) {
    const auto stream = ctx.stream();
    if (side[0] == 0) return;
    if (side[0] == 1) {
        fwht(data, count, side[1], stream);
        return;
    }
    const int shards = side[2];
    side_layout layout{width, width / shards, nparts ? nparts : 1, {}, {}};
    for (int i = 0, offset = 0; i < layout.parts; ++i) {
        const int size = nparts ? parts[i] : width;
        layout.starts[i] = offset;
        layout.sizes[i] = size / shards;
        offset += size;
    }
    ggml_cuda_pool_alloc<float> packed(ctx.pool(), count);
    const int m = ggml_trellis_hadfull_m(layout.slice);
    const int run = layout.slice / m;
    pack_side<<<blocks(count), threads, 0, stream>>>(data, packed.ptr, layout, false, count);
    if (m == 1) {
        fwht(packed.ptr, count, run, stream);
        pack_side<<<blocks(count), threads, 0, stream>>>(packed.ptr, data, layout, true, count);
        return;
    }
    if (!inverse) fwht(packed.ptr, count, run, stream);
    paley<<<blocks(count), threads, 0, stream>>>(packed.ptr, data, make_signs(m, inverse), m, run,
                                               float(1.0 / std::sqrt(double(m))), count);
    if (inverse) fwht(data, count, run, stream);
    CUDA_CHECK(cudaMemcpyAsync(packed.ptr, data, count * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    pack_side<<<blocks(count), threads, 0, stream>>>(packed.ptr, data, layout, true, count);
}
} // namespace

bool ggml_cuda_trellis_supported(const ggml_tensor * op) {
    const auto * w = op->src[0];
    const auto * x = op->src[1];
    if (!w || !x || !ggml_is_trellis_type(w->type) || !ggml_is_matrix(w)) {
        return false;
    }
    const int64_t k = w->ne[0];
    const int64_t n = w->ne[1];
    if (k <= 0 || n <= 0 || k > INT32_MAX || n > INT32_MAX || !ggml_is_contiguous(w)) {
        return false;
    }
    if (x->type != GGML_TYPE_F32 || x->ne[0] != k || x->nb[0] != sizeof(float) || x->nb[1] % sizeof(float) != 0 ||
        op->type != GGML_TYPE_F32 || op->ne[0] != n) {
        return false;
    }

    // The kernels flatten the input rows and use one-dimensional launch grids.
    const int64_t max_rows = INT32_MAX / std::max(k, n);
    int64_t rows = 1;
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (x->ne[i] <= 0 || x->ne[i] > max_rows / rows || op->ne[i] != x->ne[i] ||
            x->nb[1] > SIZE_MAX / rows || x->nb[i] != x->nb[1] * rows) {
            return false;
        }
        rows *= x->ne[i];
    }
    if ((rows > 1 && x->nb[1] > (SIZE_MAX - k * sizeof(float)) / (rows - 1)) ||
        (rows >= gemm_min_tokens && k > INT32_MAX / std::min<int64_t>(n, gemm_stripe_rows))) {
        return false;
    }
    const auto * d = reinterpret_cast<const int32_t *>(op->op_params);
    if (!ggml_is_contiguous(op) || d[6] < 0 || d[6] > GGML_TRELLIS_MAX_PARTS ||
        !ggml_trellis_desc_supported(k, n, d, 7 + d[6])) {
        return false;
    }

    const ggml_type types[] = {GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_I32, GGML_TYPE_I32};
    const int64_t lengths[] = {k / 16, k, n, k, n};
    for (int i = 0; i < 5; ++i) {
        const auto * t = op->src[i + 2];
        if (!t) {
            if (i < 3) {
                return false;
            }
        } else if (t->type != types[i] || !ggml_is_vector(t) || t->ne[0] != lengths[i] || !ggml_is_contiguous(t)) {
            return false;
        }
    }
    return true;
}

void ggml_cuda_op_trellis(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (ggml_cuda_info().devices[ctx.device].cc == GGML_TRELLIS_HIP_GFX1201 && trellis_hip_rdna4::supports(dst)) {
        trellis_hip_rdna4::mul_mat(ctx, dst);
        return;
    }
    const auto * w = dst->src[0];
    const auto * x = dst->src[1];
    const auto * d = reinterpret_cast<const int32_t *>(dst->op_params);
    const auto * pin = dst->src[5];
    const auto * pout = dst->src[6];
    const int k = w->ne[0], n = w->ne[1];
    const size_t t = ggml_nrows(x), nx = t * k, ny = t * n;
    const auto stream = ctx.stream();
    ggml_cuda_pool_alloc<float> xr(ctx.pool(), nx), yr(ctx.pool(), ny);
    scale_gather<<<blocks(nx), threads, 0, stream>>>(
        static_cast<const float *>(x->data), xr.ptr, nullptr,
        pin ? static_cast<const int32_t *>(pin->data) : nullptr, k, x->nb[1] / sizeof(float), nx);
    scale_gather<<<blocks(nx), threads, 0, stream>>>(
        xr.ptr, xr.ptr, static_cast<const float *>(dst->src[3]->data), nullptr, k, k, nx);
    apply_side(ctx, xr.ptr, k, nx, d + 3, nullptr, 0, false);
    fold_scales<<<blocks(nx), threads, 0, stream>>>(xr.ptr, static_cast<const half *>(dst->src[2]->data), k, nx);
    const auto * codes = static_cast<const uint32_t *>(w->data);
    switch (w->type) {
        case GGML_TYPE_TRL1: product<1>(ctx, codes, xr.ptr, yr.ptr, k, n, t); break;
        case GGML_TYPE_TRL2: product<2>(ctx, codes, xr.ptr, yr.ptr, k, n, t); break;
        case GGML_TYPE_TRL3: product<3>(ctx, codes, xr.ptr, yr.ptr, k, n, t); break;
        case GGML_TYPE_TRL4: product<4>(ctx, codes, xr.ptr, yr.ptr, k, n, t); break;
        case GGML_TYPE_TRL5: product<5>(ctx, codes, xr.ptr, yr.ptr, k, n, t); break;
        case GGML_TYPE_TRL6: product<6>(ctx, codes, xr.ptr, yr.ptr, k, n, t); break;
        default: GGML_ABORT("unsupported HIP trellis type");
    }
    apply_side(ctx, yr.ptr, n, ny, d, d + 7, d[6], true);
    scale_gather<<<blocks(ny), threads, 0, stream>>>(
        yr.ptr, static_cast<float *>(dst->data), static_cast<const float *>(dst->src[4]->data),
        pout ? static_cast<const int32_t *>(pout->data) : nullptr, n, n, ny);
    CUDA_CHECK(cudaGetLastError());
}
