// SPDX-License-Identifier: Apache-2.0
// RDNA4 Trellis kernels.
#pragma once

#include "mma.cuh"

#include <cmath>
#include <cstdlib>

namespace trellis_hip_rdna4 {

// tokens per chunk; scratch buffers are reused for every chunk
constexpr int BATCH = 512;
// largest blockwise Hadamard done in one workgroup's LDS (4096 floats = 16 KiB)
constexpr int BLOCK_LDS_MAX = 4096;

struct op_t {
    const ggml_tensor * w, * x, * s16, * pre, * post, * pin, * pout;
    int rows_kind, rows_block, rows_shards, cols_kind, cols_block, cols_shards, nparts;
    const int32_t * parts;
    int K, N, bits;
    int64_t T;

    explicit op_t(const ggml_tensor * dst) {
        w    = dst->src[0];
        x    = dst->src[1];
        s16  = dst->src[2];
        pre  = dst->src[3];
        post = dst->src[4];
        pin  = dst->src[5];
        pout = dst->src[6];
        const int32_t * d = (const int32_t *) dst->op_params;
        rows_kind   = d[0];
        rows_block  = d[1];
        rows_shards = d[2];
        cols_kind   = d[3];
        cols_block  = d[4];
        cols_shards = d[5];
        nparts      = d[6];
        parts       = d + 7;
        K    = (int) w->ne[0];
        N    = (int) w->ne[1];
        T    = x->ne[1] * x->ne[2] * x->ne[3];
        bits = (int) w->type - (int) GGML_TYPE_TRL1 + 1;
    }
};

// The 16-bit trellis state of ring position p: the window of the tile's circular 256*bits stream
// ending at bit (p+1)*bits. Stream bit q is bit 31 - q%32 of word q/32, so the window is taken
// MSB-first from the word pair (hi, lo).
template <int BITS>
static __device__ __forceinline__ uint32_t state_at(const uint32_t * tile, int word, int off) {
    constexpr int NW = 8 * BITS;
    const uint32_t hi = tile[word], lo = tile[(word + 1) % NW];
    return (off <= 16 ? (hi >> (16 - off)) : (hi << (off - 16)) | (lo >> (48 - off))) & 0xFFFFu;
}

// the INT8 grid: x = state * mult; v = sum of the four 6-bit digits of x, minus 126.
// Each digit is 0..63, so the byte fields of (x >> 2) are 0..63 and their sum is <= 252: the
// multiply by 0x01010101 accumulates them into the top byte without carries. Exact, |v| <= 126.
//
// This must be an UNSIGNED byte sum. ggml_cuda_dp4a() is the signed form and would read digits
// >= 32 as negative (libtrellis uses the unsigned __dp4a overload for the same reason).
static __device__ __forceinline__ int grid_int(uint32_t st) {
#if defined(AMD_WMMA_128B_AVAILABLE)
    // st is 16-bit, so splitting 0x83DCD12D lets both halves use full-rate 24-bit multiply
    const uint32_t lo = __umul24(st, 0xD12Du);
    const uint32_t hi = __umul24(st, 0x83DCu);
    const uint32_t x = lo + (hi << 16);
    const uint32_t digits = (x >> 2) & 0x3F3F3F3Fu;
    return (int) __builtin_amdgcn_sad_u8(digits, 0u, 0u) - 126;
#else
    const uint32_t x = st * 0x83DCD12Du;
    return int(((x >> 2) & 63) + ((x >> 10) & 63) + ((x >> 18) & 63) + ((x >> 26) & 63)) - 126;
#endif
}

static __device__ __forceinline__ float grid_value(uint32_t st) {
    return (float) grid_int(st);
}

// ring position of element (r, c) within a 16x16 tile; the inverse of the reference map
// p = 8a + j -> row = a/4 + 8*((j>>2)&1), col = 2*(a%4) + (j&1) + 8*((j>>1)&1)
static __device__ __forceinline__ int ring_pos(int r, int c) {
    return 8 * (4 * (r % 8) + (c % 8) / 2) + 4 * (r / 8) + 2 * (c / 8) + c % 2;
}

struct side_args {
    int n;        // entries per row (K or N)
    int len;      // block size (kind 1) or slice length (kind 2)
    int run;      // len / m
    int m;        // Paley factor: 1, 12, 20, 68
    int h;        // butterfly stage
    int shards;   // kind 1: n/block; kind 2: the descriptor's shards
    int nparts;   // kind 2 fused parts, else 0
    int inverse;
    int parts[GGML_TRELLIS_MAX_PARTS];
    float scale;
};

// the stored index of slice `shard`'s local entry l (fused parts are stored part by part)
static __device__ __forceinline__ int64_t stored(const side_args & a, int l, int shard) {
    if (a.nparts == 0) {
        return (int64_t) shard * a.len + l;
    }
    int64_t start = 0;
    for (int p = 0; p < a.nparts; ++p) {
        const int size = a.parts[p] / a.shards;
        if (l < size) {
            return start + (int64_t) shard * size + l;
        }
        l -= size;
        start += a.parts[p];
    }
    return start;
}

// quadratic character mod q (q = m-1 is prime for m in 12, 20, 68)
static __device__ __forceinline__ int chi(int d, int q) {
    d = (d % q + q) % q;
    if (d == 0) {
        return 0;
    }
    for (int i = 1; i < q; ++i) {
        if (i * i % q == d) {
            return 1;
        }
    }
    return -1;
}

// Wave-shuffle + register Hadamard for 128 elements (32 lanes x 4 elements: k = lane + 32*e).
static __device__ __forceinline__ void hadamard_128_warp(float (&v)[4], int lane) {
#pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        const bool hi = (lane & h) != 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float o = ggml_cuda_shfl_xor(v[e], h);
            v[e] = hi ? (o - v[e]) : (v[e] + o);
        }
    }
    {
        const float a0 = v[0], b0 = v[1], a1 = v[2], b1 = v[3];
        v[0] = a0 + b0; v[1] = a0 - b0;
        v[2] = a1 + b1; v[3] = a1 - b1;
    }
    {
        const float a0 = v[0], b0 = v[2], a1 = v[1], b1 = v[3];
        v[0] = a0 + b0; v[2] = a0 - b0;
        v[1] = a1 + b1; v[3] = a1 - b1;
    }
}

// Fuse the input gather, scale and blockwise Hadamard.
static __global__ void k_input_block_fast(
        int K, int len, float scale, int first, int ne1, int ne2, size_t nb1, size_t nb2, size_t nb3,
        const char * __restrict__ x, const float * __restrict__ pre, const int32_t * __restrict__ pin,
        float * __restrict__ dst) {
    extern __shared__ float sm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int b0 = blockIdx.x * len;
    const int64_t c = first + blockIdx.y;
    const float * row = (const float *) (x + (c % ne1) * nb1 + ((c / ne1) % ne2) * nb2 + (c / ((int64_t) ne1 * ne2)) * nb3);
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int i = warp * 128 + lane + 32 * e;
        const int k = b0 + i;
        v[e] = row[pin ? pin[k] : k] * pre[k];
    }
    hadamard_128_warp(v, lane);
    const int64_t dst_base = (int64_t) blockIdx.y * K + b0;
    if (len == 128) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            dst[dst_base + lane + 32 * e] = v[e] * scale;
        }
        return;
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        sm[warp * 128 + lane + 32 * e] = v[e];
    }
    __syncthreads();
    for (int h = 128; h < len; h <<= 1) {
        const int half = len >> 1;
        const int mask = h - 1;
        for (int j = threadIdx.x; j < half; j += blockDim.x) {
            const int i = ((j & ~mask) << 1) | (j & mask);
            const float a = sm[i], b = sm[i + h];
            sm[i]     = a + b;
            sm[i + h] = a - b;
        }
        __syncthreads();
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int i = warp * 128 + lane + 32 * e;
        dst[dst_base + i] = sm[i] * scale;
    }
}

template <bool in_place>
static __global__ void k_output_block_fast(
        int N, int len, float scale, int first, const float * __restrict__ src,
        const float * __restrict__ post, float * __restrict__ dst) {
    extern __shared__ float sm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int b0 = blockIdx.x * len;
    const int64_t src_base = (int64_t) blockIdx.y * N + b0;
    const float * input = in_place ? dst : src;
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        v[e] = input[src_base + warp * 128 + lane + 32 * e];
    }
    hadamard_128_warp(v, lane);
    const int64_t dst_base = (int64_t) (first + blockIdx.y) * N + b0;
    if (len == 128) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int i = lane + 32 * e;
            dst[dst_base + i] = v[e] * scale * (post ? post[b0 + i] : 1.0f);
        }
        return;
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        sm[warp * 128 + lane + 32 * e] = v[e];
    }
    __syncthreads();
    for (int h = 128; h < len; h <<= 1) {
        const int half = len >> 1;
        const int mask = h - 1;
        for (int j = threadIdx.x; j < half; j += blockDim.x) {
            const int i = ((j & ~mask) << 1) | (j & mask);
            const float a = sm[i], b = sm[i + h];
            sm[i]     = a + b;
            sm[i + h] = a - b;
        }
        __syncthreads();
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int i = warp * 128 + lane + 32 * e;
        dst[dst_base + i] = sm[i] * scale * (post ? post[b0 + i] : 1.0f);
    }
}

// Fused MLP middle: output Hadamard + post on gate & up, SwiGLU, pre + input Hadamard for down.
// Matches trellis_swiglu_rht in trellis.cu; blockDim.x = len / 4, smem = 2 * len * sizeof(float) for len > 128.
static __global__ void k_swiglu_block_fast(
        const float * __restrict__ g, const float * __restrict__ u, float * __restrict__ xd,
        int n, int len, float scale,
        const float * __restrict__ post_g, const float * __restrict__ post_u, const float * __restrict__ pre_d) {
    extern __shared__ float sm[];
    float * sg = sm;
    float * su = sm + len;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int b0 = blockIdx.x * len;
    const int64_t row_base = (int64_t) blockIdx.y * n + b0;
    float vg[4], vu[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int i = warp * 128 + lane + 32 * e;
        vg[e] = g[row_base + i];
        vu[e] = u[row_base + i];
    }
    hadamard_128_warp(vg, lane);
    hadamard_128_warp(vu, lane);
    if (len > 128) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int i = warp * 128 + lane + 32 * e;
            sg[i] = vg[e];
            su[i] = vu[e];
        }
        __syncthreads();
        const int half = len >> 1;
        for (int h = 128; h < len; h <<= 1) {
            const int mask = h - 1;
            for (int j = threadIdx.x; j < half; j += blockDim.x) {
                const int i = ((j & ~mask) << 1) | (j & mask);
                float a = sg[i], b = sg[i + h];
                sg[i] = a + b; sg[i + h] = a - b;
                a = su[i]; b = su[i + h];
                su[i] = a + b; su[i + h] = a - b;
            }
            __syncthreads();
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int i = warp * 128 + lane + 32 * e;
            vg[e] = sg[i];
            vu[e] = su[i];
        }
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int i = warp * 128 + lane + 32 * e;
        const float gv = vg[e] * scale * post_g[b0 + i];
        const float uv = vu[e] * scale * post_u[b0 + i];
        const float silu = gv / (1.0f + expf(-gv));
        vg[e] = (silu * uv) * pre_d[b0 + i];
    }
    hadamard_128_warp(vg, lane);
    if (len == 128) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            xd[row_base + lane + 32 * e] = vg[e] * scale;
        }
        return;
    }
    __syncthreads();
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        sg[warp * 128 + lane + 32 * e] = vg[e];
    }
    __syncthreads();
    const int half = len >> 1;
    for (int h = 128; h < len; h <<= 1) {
        const int mask = h - 1;
        for (int j = threadIdx.x; j < half; j += blockDim.x) {
            const int i = ((j & ~mask) << 1) | (j & mask);
            const float a = sg[i], b = sg[i + h];
            sg[i] = a + b; sg[i + h] = a - b;
        }
        __syncthreads();
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int i = warp * 128 + lane + 32 * e;
        xd[row_base + i] = sg[i] * scale;
    }
}

// blockwise Hadamard of `len` entries in LDS: stages h = 1, 2, 4, ... in order, then one scale
static __global__ void k_block(side_args a, float * v) {
    extern __shared__ float sm[];
    const int64_t base = (int64_t) blockIdx.y * a.n + (int64_t) blockIdx.x * a.len;
    for (int i = threadIdx.x; i < a.len; i += blockDim.x) {
        sm[i] = v[base + i];
    }
    __syncthreads();
    for (int h = 1; h < a.len; h *= 2) {
        for (int j = threadIdx.x; j < a.len / 2; j += blockDim.x) {
            const int i = (j / h) * (2 * h) + j % h;
            const float x = sm[i], y = sm[i + h];
            sm[i]     = x + y;
            sm[i + h] = x - y;
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < a.len; i += blockDim.x) {
        v[base + i] = sm[i] * a.scale;
    }
}

// one butterfly stage over global memory; the last stage carries the run's normalisation
static __global__ void k_butterfly(side_args a, float * v) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= a.len / 2) {
        return;
    }
    const int l = (idx / a.h) * (2 * a.h) + idx % a.h;
    const int64_t j = (int64_t) blockIdx.z * a.n + stored(a, l,       blockIdx.y);
    const int64_t k = (int64_t) blockIdx.z * a.n + stored(a, l + a.h, blockIdx.y);
    const float x = v[j], y = v[k];
    v[j] = (x + y) * a.scale;
    v[k] = (x - y) * a.scale;
}

// the Paley cross term H_m (x) I, m in 12, 20, 68
static __global__ void k_cross(side_args a, const float * __restrict__ src, float * __restrict__ dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= a.len) {
        return;
    }
    const int row = idx / a.run, j = idx % a.run;
    float sum = 0.0f;
    for (int b = 0; b < a.m; ++b) {
        const int r = a.inverse ? b : row, c = a.inverse ? row : b;
        const int s = r == c ? 1 : r == 0 ? 1 : c == 0 ? -1 : chi(c - r, a.m - 1);
        const float x = src[(int64_t) blockIdx.z * a.n + stored(a, b * a.run + j, blockIdx.y)];
        sum += s > 0 ? x : -x;
    }
    dst[(int64_t) blockIdx.z * a.n + stored(a, idx, blockIdx.y)] = sum * a.scale;
}

// dst[t*K + k] = x_row(first + t)[pin ? pin[k] : k] * pre[k]
static __global__ void k_input(int K, int first, int ne1, int ne2, size_t nb1, size_t nb2, size_t nb3,
                               const char * __restrict__ x, const float * __restrict__ pre,
                               const int32_t * __restrict__ pin, float * __restrict__ dst) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) {
        return;
    }
    const int64_t c = first + blockIdx.y;
    const char * row = x + (c % ne1) * nb1 + ((c / ne1) % ne2) * nb2 + (c / ((int64_t) ne1 * ne2)) * nb3;
    dst[(int64_t) blockIdx.y * K + k] = ((const float *) row)[pin ? pin[k] : k] * pre[k];
}

// dst[(first + t)*N + n] = y[t*N + j] * post[j], j = pout ? pout[n] : n
static __global__ void k_output(int N, int first, const float * __restrict__ y, const float * __restrict__ post,
                                const int32_t * __restrict__ pout, float * __restrict__ dst) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) {
        return;
    }
    const int j = pout ? pout[n] : n;
    dst[(int64_t) (first + blockIdx.y) * N + n] = y[(int64_t) blockIdx.y * N + j] * post[j];
}

// Decode GEMV for one token: a lane decodes the 8 ring positions p = 8a + j (j = 0..7), which are
// row a/4 at columns 2q, 2q+1, 2q+8, 2q+9 and row a/4 + 8 at the same columns (q = a%4). So one
// tile visit yields two rows x four columns, the block scale is loaded once per tile, and the
// activations are read as float2 pairs. Four waves cover the 16 rows; lane/8 gives four
// independent K partitions that are summed at the end.
// The 8 windows of positions 8a..8a+7 are consecutive, so a few 32-bit anchors cover them all:
// one anchor holds SPAN = 16/BITS + 1 consecutive windows. This is libtrellis's Plan<K> scheme
// (ggml-cuda/libtrellis/src/common.cuh); it needs ANCH word pairs instead of 8.
template <int BITS>
struct anchor_plan {
    static constexpr int NW   = 8 * BITS;
    static constexpr int TOTAL = 256 * BITS;
    static constexpr int SPAN = 16 / BITS + 1;
    static constexpr int ANCH = (8 + SPAN - 1) / SPAN;
    int w0[ANCH], w1[ANCH], sh[ANCH];

    static __device__ __forceinline__ anchor_plan make(int a) {
        anchor_plan p;
#pragma unroll
        for (int x = 0; x < ANCH; ++x) {
            const int e = ((8 * a + 7 - x * SPAN + 1) * BITS) % TOTAL;  // window end, exclusive
            p.w0[x] = ((e - 32 + TOTAL) >> 5) % NW;
            p.w1[x] = (p.w0[x] + 1) % NW;
            p.sh[x] = 32 - (e & 31);
        }
        return p;
    }
};

template <int BITS, int I>
static __device__ __forceinline__ uint32_t anchored_state(const uint32_t (&r)[anchor_plan<BITS>::ANCH]) {
    constexpr int SPAN   = anchor_plan<BITS>::SPAN;
    constexpr int A      = (7 - I) / SPAN;
    constexpr int ANCHOR = 7 - A * SPAN;
    constexpr int C      = (ANCHOR - I) * BITS;
    return (r[A] >> C) & 0xFFFFu;
}

template <int BITS>
static __device__ __forceinline__ void decode8_plan(const uint32_t * tile, const anchor_plan<BITS> & p, float * v) {
    uint32_t r[anchor_plan<BITS>::ANCH];
#pragma unroll
    for (int x = 0; x < anchor_plan<BITS>::ANCH; ++x) {
        // libtrellis's operand order: the *next* word is the low half, the anchor word the high half
        r[x] = __funnelshift_rc(tile[p.w1[x]], tile[p.w0[x]], p.sh[x]);
    }
    v[0] = grid_value(anchored_state<BITS, 0>(r));
    v[1] = grid_value(anchored_state<BITS, 1>(r));
    v[2] = grid_value(anchored_state<BITS, 2>(r));
    v[3] = grid_value(anchored_state<BITS, 3>(r));
    v[4] = grid_value(anchored_state<BITS, 4>(r));
    v[5] = grid_value(anchored_state<BITS, 5>(r));
    v[6] = grid_value(anchored_state<BITS, 6>(r));
    v[7] = grid_value(anchored_state<BITS, 7>(r));
}

template <int BITS>
static __device__ __forceinline__ void decode8(const uint32_t * tile, int a, float * v) {
    decode8_plan<BITS>(tile, anchor_plan<BITS>::make(a), v);
}

template <int BITS, int COLS>
static __global__ void __launch_bounds__(128) k_product_mv(
        const uint32_t * __restrict__ codes, const half * __restrict__ s16,
        const float * __restrict__ x, float * __restrict__ y, int K, int N, int batch) {
    constexpr int NW = 8 * BITS;
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int tile_row = blockIdx.x;
    const int col0 = COLS * blockIdx.y;
    const int q = lane % 4;
    const int c = 2 * q;
    const int a = 8 * wave + (lane % 8);
    const int row = 16 * tile_row + a / 4;
    const int kt = K / 16;
    const uint32_t * tiles = codes + (int64_t) tile_row * kt * NW;

    float4 s0[COLS], s1[COLS];
#pragma unroll
    for (int t = 0; t < COLS; ++t) {
        s0[t] = make_float4(0.f, 0.f, 0.f, 0.f);
        s1[t] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    for (int i = lane / 8; i < kt; i += 4) {
        const uint32_t * tile = tiles + (int64_t) i * NW;
        float v[8];
        decode8<BITS>(tile, a, v);
        const float s = __half2float(s16[i]);
        const float w0 = v[0] * s, w1 = v[1] * s, w2 = v[2] * s, w3 = v[3] * s;
        const float w4 = v[4] * s, w5 = v[5] * s, w6 = v[6] * s, w7 = v[7] * s;
#pragma unroll
        for (int t = 0; t < COLS; ++t) {
            if (col0 + t < batch) {
                const float * xt = x + (int64_t) (col0 + t) * K + 16 * i + c;
                const float2 x0 = *reinterpret_cast<const float2 *>(xt);
                const float2 x1 = *reinterpret_cast<const float2 *>(xt + 8);
                s0[t].x += w0 * x0.x;
                s0[t].y += w1 * x0.y;
                s0[t].z += w2 * x1.x;
                s0[t].w += w3 * x1.y;
                s1[t].x += w4 * x0.x;
                s1[t].y += w5 * x0.y;
                s1[t].z += w6 * x1.x;
                s1[t].w += w7 * x1.y;
            }
        }
    }
    // sum the four K partitions (lane/8), then the four column groups (lane%4)
#pragma unroll
    for (int t = 0; t < COLS; ++t) {
#pragma unroll
        for (int off = 8; off <= 16; off *= 2) {
            s0[t].x += ggml_cuda_shfl_xor(s0[t].x, off);
            s0[t].y += ggml_cuda_shfl_xor(s0[t].y, off);
            s0[t].z += ggml_cuda_shfl_xor(s0[t].z, off);
            s0[t].w += ggml_cuda_shfl_xor(s0[t].w, off);
            s1[t].x += ggml_cuda_shfl_xor(s1[t].x, off);
            s1[t].y += ggml_cuda_shfl_xor(s1[t].y, off);
            s1[t].z += ggml_cuda_shfl_xor(s1[t].z, off);
            s1[t].w += ggml_cuda_shfl_xor(s1[t].w, off);
        }
#pragma unroll
        for (int off = 1; off <= 2; off *= 2) {
            s0[t].x += ggml_cuda_shfl_xor(s0[t].x, off);
            s0[t].y += ggml_cuda_shfl_xor(s0[t].y, off);
            s0[t].z += ggml_cuda_shfl_xor(s0[t].z, off);
            s0[t].w += ggml_cuda_shfl_xor(s0[t].w, off);
            s1[t].x += ggml_cuda_shfl_xor(s1[t].x, off);
            s1[t].y += ggml_cuda_shfl_xor(s1[t].y, off);
            s1[t].z += ggml_cuda_shfl_xor(s1[t].z, off);
            s1[t].w += ggml_cuda_shfl_xor(s1[t].w, off);
        }
        if (lane < 8 && q == 0 && col0 + t < batch) {
            y[(int64_t) (col0 + t) * N + row]     = s0[t].x + s0[t].y + s0[t].z + s0[t].w;
            y[(int64_t) (col0 + t) * N + row + 8] = s1[t].x + s1[t].y + s1[t].z + s1[t].w;
        }
    }
}

//
// A workgroup is 4 waves and covers 64 output rows x 16 tokens per k-block of 16. Each wave owns one
// 16x16 weight tile and decodes it with the same decode8 as the scalar path, but into LDS as exact
// integer grid values (all of [-126, 126] are exact in f16). The block scale s16[k/16] is folded into
// the activations instead, once per block, so the B operand stays a plain f16 matrix of tokens.
// Result mapping: mma(D, A, B) puts the B row index in get_i and the A row index in
// get_j, i.e. D[get_i][get_j] = sum_k A[get_j][k] * B[get_i][k].
//
// Launch gated at runtime on RDNA4 (validated there only); other targets keep the scalar GEMV.

// The gfx12 fragment layouts in mma.cuh are expressed in terms of threadIdx.x, i.e. they assume one
// wave per workgroup. These helpers take the lane index explicitly so four waves can share a workgroup.
// Layout: A/B element (i, j) sits at i = lane % 16, j = 4*(lane/16) + l (half2 units); C element (i, j)
// at i = lane % 16, j = 8*(lane/16) + l (float units).
#if defined(AMD_WMMA_128B_AVAILABLE)
template <int STRIDE_H2, ggml_cuda_mma::data_layout dl>
static __device__ __forceinline__ void mm_load_ab_s(ggml_cuda_mma::tile<16, 8, half2, dl> & t,
                                                    const half2 * __restrict__ p, int lane) {
    const int i = lane % 16, j0 = 4 * (lane / 16);
    __builtin_memcpy(t.x, __builtin_assume_aligned(p + i * STRIDE_H2 + j0, 16), sizeof(t.x));
}

template <ggml_cuda_mma::data_layout dl>
static __device__ __forceinline__ void mm_load_ab(ggml_cuda_mma::tile<16, 8, half2, dl> & t,
                                                 const half2 * __restrict__ p, int lane) {
    mm_load_ab_s<8, dl>(t, p, lane);
}

static __device__ __forceinline__ bool outside_half_range(float value) {
    return (__float_as_uint(value) & 0x7fffffffu) > __float_as_uint(65504.0f);
}

template <int width>
static __device__ __forceinline__ unsigned range_mask(unsigned mask) {
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        mask |= ggml_cuda_shfl_xor(mask, offset);
    }
    return mask;
}

template <int bits>
static __device__ __attribute__((noinline)) float product_row_f32(
        const uint32_t * codes, const half * s16, const float * input, int K, int row) {
    constexpr int words = 8 * bits;
    constexpr int ring_bits = 256 * bits;
    const int kt = K / 16;
    const uint32_t * tiles = codes + (int64_t) (row / 16) * kt * words;
    float sum = 0.0f;
    for (int block = 0; block < kt; ++block) {
        const float scale = __half2float(s16[block]);
        const uint32_t * tile = tiles + (int64_t) block * words;
#pragma unroll
        for (int column = 0; column < 16; ++column) {
            const int pos = ring_pos(row % 16, column);
            const int start = ((pos + 1) * bits + ring_bits - 16) % ring_bits;
            const float weight = grid_value(state_at<bits>(tile, start / 32, start % 32)) * scale;
            sum += weight * input[block * 16 + column];
        }
    }
    return sum;
}

template <int bits>
static __device__ __forceinline__ void store_product(
        const uint32_t * codes, const half * s16, const float * x, float * y,
        int K, int N, int token, int row, const float (&values)[8], bool use_f32) {
    float * dst = y + (int64_t) token * N + row;
    if (use_f32) {
        const float * input = x + (int64_t) token * K;
        for (int i = 0; i < 8; ++i) {
            dst[i] = product_row_f32<bits>(codes, s16, input, K, row + i);
        }
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            dst[i] = values[i];
        }
    }
}
#endif

// 128 rows x 128 tokens workgroup (2x4 wave grid, 8 waves = 256 threads).
// Each wave computes 4x2 tiles of 16x16: every B (activation) LDS load is reused across 4 M-tiles,
// and 4 independent accumulators d[0..3] hide WMMA pipeline latency.
// sx row stride is padded to 12 half2 (48B = 12 banks) to cut LDS bank conflicts; total LDS = 32 KiB (2 WGs/CU).
template <int BITS, int KG>
static __global__ void __launch_bounds__(256, 2) k_product_mm_128x128(
        const uint32_t * __restrict__ codes, const half * __restrict__ s16,
        const float * __restrict__ x, float * __restrict__ y, int K, int N, int batch) {
#if defined(AMD_WMMA_128B_AVAILABLE)
    constexpr int NW = 8 * BITS;
    constexpr int SX_STRIDE = 12;
    constexpr ggml_cuda_mma::data_layout layout = ggml_cuda_mma::get_input_data_layout();

    __shared__ __align__(16) half2 sw[8][KG][16 * 8];              // A tiles: 8 m-tiles, k-tile (8 KiB for KG=2)
    __shared__ __align__(16) half2 sx[KG][8][2][16 * SX_STRIDE];   // B tiles: k-tile, 8 sub-tiles, hi/lo (24 KiB for KG=2)

    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm   = wave / 4, wn = wave % 4;               // 2 M-waves x 4 N-waves
    const int tile_base = 8 * blockIdx.x;
    const int col0 = 128 * blockIdx.y;
    const int kt   = K / 16;

    // Weight decode: 8 waves cover the 8 M-tiles (1 M-tile per wave, all KG k-tiles)
    const int decode_tile = tile_base + wave;
    const bool decode_live = 16 * decode_tile < N;

    const int m0 = 4 * wm;

    const int c2 = lane % 4;
    const int r  = lane / 4;
    const anchor_plan<BITS> plan = anchor_plan<BITS>::make(lane);

    const int t0  = threadIdx.x / 4;        // 0..63
    const int kp  = (threadIdx.x % 4) * 4;  // 0, 4, 8, 12 (float index)
    const int kp2 = (threadIdx.x % 4) * 2;  // 0, 2, 4, 6 (half2 index)

    float acc[4][2][8] = {};
    unsigned overflow = 0;

    for (int kb0 = 0; kb0 < kt; kb0 += KG) {
        const int jn = min(KG, kt - kb0);
        if (decode_live) {
            const uint32_t * decode_tiles = codes + (int64_t) decode_tile * kt * NW;
#pragma unroll
            for (int j = 0; j < KG; ++j) {
                if (j >= jn) { break; }
                half2 * w2 = sw[wave][j];
                float v[8];
                decode8_plan<BITS>(decode_tiles + (int64_t) (kb0 + j) * NW, plan, v);
                w2[r * 8 + c2 + 0]       = __floats2half2_rn(v[0], v[1]);
                w2[r * 8 + c2 + 4]       = __floats2half2_rn(v[2], v[3]);
                w2[(r + 8) * 8 + c2 + 0] = __floats2half2_rn(v[4], v[5]);
                w2[(r + 8) * 8 + c2 + 4] = __floats2half2_rn(v[6], v[7]);
            }
        } else {
#pragma unroll
            for (int j = 0; j < KG; ++j) {
                if (j >= jn) { break; }
                half2 * w2 = sw[wave][j];
                const half2 zero = __floats2half2_rn(0.0f, 0.0f);
                w2[r * 8 + c2 + 0]       = zero;
                w2[r * 8 + c2 + 4]       = zero;
                w2[(r + 8) * 8 + c2 + 0] = zero;
                w2[(r + 8) * 8 + c2 + 4] = zero;
            }
        }
#pragma unroll
        for (int j = 0; j < KG; ++j) {
            if (j >= jn) { break; }
            const float s = __half2float(s16[kb0 + j]);
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int tok = p * 64 + t0;
                const int sub = tok >> 4;
                const int tr  = tok & 15;
                const int col = col0 + tok;
                const float4 fv = col < batch
                    ? *reinterpret_cast<const float4 *>(x + (int64_t) col * K + 16 * (kb0 + j) + kp)
                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                float f0 = fv.x * s, f1 = fv.y * s, f2 = fv.z * s, f3 = fv.w * s;
                if (outside_half_range(f0) || outside_half_range(f1) || outside_half_range(f2) || outside_half_range(f3)) {
                    overflow |= 1u << p;
                    f0 = f1 = f2 = f3 = 0.0f;
                }
                const half2 hi01 = __floats2half2_rn(f0, f1);
                const half2 hi23 = __floats2half2_rn(f2, f3);
                const float2 hf01 = __half22float2(hi01);
                const float2 hf23 = __half22float2(hi23);
                half2 * hi = sx[j][sub][0];
                half2 * lo = sx[j][sub][1];
                hi[tr * SX_STRIDE + kp2 + 0] = hi01;
                hi[tr * SX_STRIDE + kp2 + 1] = hi23;
                lo[tr * SX_STRIDE + kp2 + 0] = __floats2half2_rn(f0 - hf01.x, f1 - hf01.y);
                lo[tr * SX_STRIDE + kp2 + 1] = __floats2half2_rn(f2 - hf23.x, f3 - hf23.y);
            }
        }
        __syncthreads();

        ggml_cuda_mma::tile<16, 8, half2, layout> a[4][KG];
#pragma unroll
        for (int j = 0; j < KG; ++j) {
            if (j >= jn) { break; }
#pragma unroll
            for (int tm = 0; tm < 4; ++tm) {
                mm_load_ab_s<8, layout>(a[tm][j], sw[m0 + tm][j], lane);
            }
        }
#pragma unroll
        for (int tn = 0; tn < 2; ++tn) {
            const int sub = 2 * wn + tn;
            ggml_cuda_mma::tile<16, 16, float, ggml_cuda_mma::DATA_LAYOUT_I_MAJOR> d[4];
#pragma unroll
            for (int tm = 0; tm < 4; ++tm) {
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    d[tm].x[l] = 0.0f;
                }
            }
#pragma unroll
            for (int j = 0; j < KG; ++j) {
                if (j >= jn) { break; }
                ggml_cuda_mma::tile<16, 8, half2, layout> b_hi, b_lo;
                mm_load_ab_s<SX_STRIDE, layout>(b_hi, sx[j][sub][0], lane);
                mm_load_ab_s<SX_STRIDE, layout>(b_lo, sx[j][sub][1], lane);
#pragma unroll
                for (int tm = 0; tm < 4; ++tm) {
                    ggml_cuda_mma::mma(d[tm], a[tm][j], b_hi);
                }
#pragma unroll
                for (int tm = 0; tm < 4; ++tm) {
                    ggml_cuda_mma::mma(d[tm], a[tm][j], b_lo);
                }
            }
#pragma unroll
            for (int tm = 0; tm < 4; ++tm) {
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    acc[tm][tn][l] += d[tm].x[l];
                }
            }
        }
        __syncthreads();
    }

    // Reuse the weight tile storage after the last barrier.
    overflow = range_mask<4>(overflow);
    if ((threadIdx.x % 4) == 0) {
        sw[0][0][t0] = __floats2half2_rn(float(overflow & 1u), float(overflow >> 1));
    }
    __syncthreads();

#pragma unroll
    for (int tm = 0; tm < 4; ++tm) {
        const int row_base = 16 * (tile_base + m0 + tm);
        if (row_base < N) {
            const int crow = row_base + 8 * (lane / 16);
#pragma unroll
            for (int tn = 0; tn < 2; ++tn) {
                const int ccol = col0 + (2 * wn + tn) * 16 + (lane & 15);
                if (ccol < batch) {
                    const int token = ccol - col0;
                    const float2 flags = __half22float2(sw[0][0][token % 64]);
                    const bool use_f32 = (token < 64 ? flags.x : flags.y) != 0.0f;
                    store_product<BITS>(codes, s16, x, y, K, N, ccol, crow, acc[tm][tn], use_f32);
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(codes, s16, x, y, K, N, batch);
    NO_DEVICE_CODE;
#endif
}

// 64 rows x 128 tokens workgroup (4x2 wave grid, 8 waves = 256 threads) for small N (< 128).
template <int BITS, int KG>
static __global__ void __launch_bounds__(256, 2) k_product_mm_64x128(
        const uint32_t * __restrict__ codes, const half * __restrict__ s16,
        const float * __restrict__ x, float * __restrict__ y, int K, int N, int batch) {
#if defined(AMD_WMMA_128B_AVAILABLE)
    constexpr int NW = 8 * BITS;
    constexpr ggml_cuda_mma::data_layout layout = ggml_cuda_mma::get_input_data_layout();

    __shared__ __align__(16) half2 sw[4][KG][16 * 8];
    __shared__ __align__(16) half2 sx[KG][8][2][16 * 8];

    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm   = wave / 2, wn = wave % 2;
    const int tile = 4 * blockIdx.x + wm;
    const int row0 = 16 * tile;
    const int col0 = 128 * blockIdx.y;
    const int kt   = K / 16;
    const bool live = row0 < N;

    const int c2 = lane % 4;
    const int r  = lane / 4;
    const anchor_plan<BITS> plan = anchor_plan<BITS>::make(lane);

    const int t0  = threadIdx.x / 4;        // 0..63
    const int kp  = (threadIdx.x % 4) * 4;  // 0, 4, 8, 12 (float index)
    const int kp2 = (threadIdx.x % 4) * 2;  // 0, 2, 4, 6 (half2 index)

    float acc[4][8] = {};
    unsigned overflow = 0;

    for (int kb0 = 0; kb0 < kt; kb0 += KG) {
        const int jn = min(KG, kt - kb0);
        if (live) {
            const uint32_t * tiles = codes + (int64_t) tile * kt * NW;
#pragma unroll
            for (int j = wn; j < KG; j += 2) {
                if (j >= jn) { break; }
                half2 * w2 = sw[wm][j];
                float v[8];
                decode8_plan<BITS>(tiles + (int64_t) (kb0 + j) * NW, plan, v);
                w2[r * 8 + c2 + 0]       = __floats2half2_rn(v[0], v[1]);
                w2[r * 8 + c2 + 4]       = __floats2half2_rn(v[2], v[3]);
                w2[(r + 8) * 8 + c2 + 0] = __floats2half2_rn(v[4], v[5]);
                w2[(r + 8) * 8 + c2 + 4] = __floats2half2_rn(v[6], v[7]);
            }
        }
#pragma unroll
        for (int j = 0; j < KG; ++j) {
            if (j >= jn) { break; }
            const float s = __half2float(s16[kb0 + j]);
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int tok = p * 64 + t0;
                const int sub = tok >> 4;
                const int tr  = tok & 15;
                const int col = col0 + tok;
                const float4 fv = col < batch
                    ? *reinterpret_cast<const float4 *>(x + (int64_t) col * K + 16 * (kb0 + j) + kp)
                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                float f0 = fv.x * s, f1 = fv.y * s, f2 = fv.z * s, f3 = fv.w * s;
                if (outside_half_range(f0) || outside_half_range(f1) || outside_half_range(f2) || outside_half_range(f3)) {
                    overflow |= 1u << p;
                    f0 = f1 = f2 = f3 = 0.0f;
                }
                const half2 hi01 = __floats2half2_rn(f0, f1);
                const half2 hi23 = __floats2half2_rn(f2, f3);
                const float2 hf01 = __half22float2(hi01);
                const float2 hf23 = __half22float2(hi23);
                half2 * hi = sx[j][sub][0];
                half2 * lo = sx[j][sub][1];
                hi[tr * 8 + kp2 + 0] = hi01;
                hi[tr * 8 + kp2 + 1] = hi23;
                lo[tr * 8 + kp2 + 0] = __floats2half2_rn(f0 - hf01.x, f1 - hf01.y);
                lo[tr * 8 + kp2 + 1] = __floats2half2_rn(f2 - hf23.x, f3 - hf23.y);
            }
        }
        __syncthreads();

        if (live) {
            ggml_cuda_mma::tile<16, 8, half2, layout> a[KG];
#pragma unroll
            for (int j = 0; j < KG; ++j) {
                if (j >= jn) { break; }
                mm_load_ab<layout>(a[j], sw[wm][j], lane);
            }
#pragma unroll
            for (int tn = 0; tn < 4; ++tn) {
                const int sub = 4 * wn + tn;
                ggml_cuda_mma::tile<16, 16, float, ggml_cuda_mma::DATA_LAYOUT_I_MAJOR> d;
#pragma unroll
                for (int l = 0; l < d.ne; ++l) {
                    d.x[l] = 0.0f;
                }
#pragma unroll
                for (int j = 0; j < KG; ++j) {
                    if (j >= jn) { break; }
                    ggml_cuda_mma::tile<16, 8, half2, layout> b_hi, b_lo;
                    mm_load_ab<layout>(b_hi, sx[j][sub][0], lane);
                    mm_load_ab<layout>(b_lo, sx[j][sub][1], lane);
                    ggml_cuda_mma::mma(d, a[j], b_hi);
                    ggml_cuda_mma::mma(d, a[j], b_lo);
                }
#pragma unroll
                for (int l = 0; l < d.ne; ++l) {
                    acc[tn][l] += d.x[l];
                }
            }
        }
        __syncthreads();
    }

    // Reuse the weight tile storage after the last barrier.
    overflow = range_mask<4>(overflow);
    if ((threadIdx.x % 4) == 0) {
        sw[0][0][t0] = __floats2half2_rn(float(overflow & 1u), float(overflow >> 1));
    }
    __syncthreads();

    if (!live) {
        return;
    }

    const int crow = row0 + 8 * (lane / 16);
#pragma unroll
    for (int tn = 0; tn < 4; ++tn) {
        const int ccol = col0 + (4 * wn + tn) * 16 + (lane & 15);
        if (ccol < batch) {
            const int token = ccol - col0;
            const float2 flags = __half22float2(sw[0][0][token % 64]);
            const bool use_f32 = (token < 64 ? flags.x : flags.y) != 0.0f;
            store_product<BITS>(codes, s16, x, y, K, N, ccol, crow, acc[tn], use_f32);
        }
    }
#else
    GGML_UNUSED_VARS(codes, s16, x, y, K, N, batch);
    NO_DEVICE_CODE;
#endif
}

// One wave per output tile. Flush each K group to FP32 to limit WMMA accumulator bias.
template <int BITS, int KG>
static __global__ void __launch_bounds__(128) k_product_mm_small(
        const uint32_t * __restrict__ codes, const half * __restrict__ s16,
        const float * __restrict__ x, float * __restrict__ y, int K, int N, int batch) {
#if defined(AMD_WMMA_128B_AVAILABLE)
    constexpr int NW = 8 * BITS;
    constexpr auto layout = ggml_cuda_mma::get_input_data_layout();

    __shared__ __align__(16) half2 sw[4][KG][16 * 8];
    __shared__ __align__(16) half2 sx[KG][2][16 * 8];

    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int tile = 4 * blockIdx.x + wave;
    const int row0 = 16 * tile;
    const int col0 = 16 * blockIdx.y;
    const int kt = K / 16;
    const bool live = row0 < N;
    const int c2 = lane % 4, r = lane / 4;
    const int t = threadIdx.x / 8, kp = threadIdx.x % 8;
    float acc[8] = {};
    unsigned overflow = 0;

    for (int kb0 = 0; kb0 < kt; kb0 += KG) {
        const int jn = min(KG, kt - kb0);
        if (live) {
            const uint32_t * tiles = codes + (int64_t) tile * kt * NW;
#pragma unroll
            for (int j = 0; j < KG; ++j) {
                if (j >= jn) { break; }
                half2 * w = sw[wave][j];
                float v[8];
                decode8<BITS>(tiles + (int64_t) (kb0 + j) * NW, lane, v);
                w[r * 8 + c2 + 0]       = __floats2half2_rn(v[0], v[1]);
                w[r * 8 + c2 + 4]       = __floats2half2_rn(v[2], v[3]);
                w[(r + 8) * 8 + c2 + 0] = __floats2half2_rn(v[4], v[5]);
                w[(r + 8) * 8 + c2 + 4] = __floats2half2_rn(v[6], v[7]);
            }
        }
#pragma unroll
        for (int j = 0; j < KG; ++j) {
            if (j >= jn) { break; }
            const float s = __half2float(s16[kb0 + j]);
            const int col = col0 + t;
            const float2 fv = col < batch
                ? *reinterpret_cast<const float2 *>(x + (int64_t) col * K + 16 * (kb0 + j) + 2 * kp)
                : make_float2(0.0f, 0.0f);
            float f0 = fv.x * s, f1 = fv.y * s;
            if (outside_half_range(f0) || outside_half_range(f1)) {
                overflow = 1;
                f0 = f1 = 0.0f;
            }
            const half2 hi = __floats2half2_rn(f0, f1);
            const float2 rounded = __half22float2(hi);
            sx[j][0][t * 8 + kp] = hi;
            sx[j][1][t * 8 + kp] = __floats2half2_rn(f0 - rounded.x, f1 - rounded.y);
        }
        __syncthreads();

        if (live) {
            ggml_cuda_mma::tile<16, 16, float, ggml_cuda_mma::DATA_LAYOUT_I_MAJOR> d;
#pragma unroll
            for (int l = 0; l < d.ne; ++l) { d.x[l] = 0.0f; }
#pragma unroll
            for (int j = 0; j < KG; ++j) {
                if (j >= jn) { break; }
                ggml_cuda_mma::tile<16, 8, half2, layout> a, b_hi, b_lo;
                mm_load_ab<layout>(a, sw[wave][j], lane);
                mm_load_ab<layout>(b_hi, sx[j][0], lane);
                mm_load_ab<layout>(b_lo, sx[j][1], lane);
                ggml_cuda_mma::mma(d, a, b_hi);
                ggml_cuda_mma::mma(d, a, b_lo);
            }
#pragma unroll
            for (int l = 0; l < d.ne; ++l) { acc[l] += d.x[l]; }
        }
        __syncthreads();
    }

    overflow = range_mask<8>(overflow);
    if (kp == 0) {
        sw[0][0][t] = __floats2half2_rn(float(overflow), 0.0f);
    }
    __syncthreads();

    const int ccol = col0 + lane % 16;
    if (live && ccol < batch) {
        const int crow = row0 + 8 * (lane / 16);
        const bool use_f32 = __half22float2(sw[0][0][lane % 16]).x != 0.0f;
        store_product<BITS>(codes, s16, x, y, K, N, ccol, crow, acc, use_f32);
    }
#else
    GGML_UNUSED_VARS(codes, s16, x, y, K, N, batch);
    NO_DEVICE_CODE;
#endif
}

static void launch_side(side_args a, float *& data, float *& tmp, int batch, bool inverse, cudaStream_t stream) {
    if (a.len <= 1) {
        return;
    }
    // small blockwise side: the whole block in one workgroup's LDS
    if (a.nparts == 0 && a.m == 1 && a.len <= BLOCK_LDS_MAX && a.len * (int) sizeof(float) <= 64 * 1024) {
        a.scale = (float) std::pow((double) a.len, -0.5);
        if (a.len >= 128) {
            const size_t smem = a.len > 128 ? (size_t) a.len * sizeof(float) : 0;
            k_output_block_fast<true><<<dim3(a.n / a.len, batch, 1), a.len / 4, smem, stream>>>(
                a.n, a.len, a.scale, 0, nullptr, nullptr, data);
        } else {
            const int threads = std::min(a.len / 2, 256);
            const dim3 grid(a.n / a.len, batch, 1);
            k_block<<<grid, std::max(threads, 1), a.len * sizeof(float), stream>>>(a, data);
        }
        return;
    }
    const dim3 block(256, 1, 1);
    auto cross = [&]() {
        if (a.m == 1) {
            return;
        }
        a.scale = (float) std::pow((double) a.m, -0.5);
        const dim3 grid((a.len + 255) / 256, a.shards, batch);
        k_cross<<<grid, block, 0, stream>>>(a, data, tmp);
        std::swap(data, tmp);
    };
    if (inverse) {
        cross();
    }
    for (a.h = 1; a.h < a.run; a.h *= 2) {
        a.scale = a.h == a.run / 2 ? (float) std::pow((double) a.run, -0.5) : 1.0f;
        const dim3 grid((a.len / 2 + 255) / 256, a.shards, batch);
        k_butterfly<<<grid, block, 0, stream>>>(a, data);
    }
    if (!inverse) {
        cross();
    }
}

// one side of the op: kind 0 identity, kind 1 blockwise, kind 2 whole-slice
static void side(float *& data, float *& tmp, int n, int batch, int kind, int block, int shards,
                 const int32_t * parts, int nparts, bool inverse, cudaStream_t stream) {
    if (kind == 0) {
        return;
    }
    side_args a = {};
    a.n       = n;
    a.len     = kind == 1 ? block : n / shards;
    a.shards  = kind == 1 ? n / block : shards;
    a.m       = kind == 1 ? 1 : ggml_trellis_hadfull_m(a.len);
    a.run     = a.len / a.m;
    a.inverse = inverse;
    a.nparts  = kind == 2 ? nparts : 0;
    for (int i = 0; i < a.nparts; ++i) {
        a.parts[i] = parts[i];
    }
    launch_side(a, data, tmp, batch, inverse, stream);
}

static bool wmma_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_TRELLIS_HIP_MM");
        return value == nullptr || std::atoi(value) != 0;
    }();
    return enabled;
}

template <int bits>
static void product(const op_t & o, const float * in, float * out, int batch, bool wmma, cudaStream_t stream) {
    const int K = o.K, N = o.N;
    const auto * codes = static_cast<const uint32_t *>(o.w->data);
    const auto * scales = static_cast<const half *>(o.s16->data);
    if (batch == 1) {
        k_product_mv<bits, 1><<<dim3(N / 16, 1, 1), 128, 0, stream>>>(codes, scales, in, out, K, N, 1);
    } else if (!wmma) {
        k_product_mv<bits, 8><<<dim3(N / 16, (batch + 7) / 8, 1), 128, 0, stream>>>(codes, scales, in, out, K, N, batch);
    } else if (batch < 32) {
        const dim3 grid((N / 16 + 3) / 4, (batch + 15) / 16, 1);
        k_product_mm_small<bits, 2><<<grid, 128, 0, stream>>>(codes, scales, in, out, K, N, batch);
    } else if (N < 128) {
        const dim3 grid((N / 16 + 3) / 4, (batch + 127) / 128, 1);
        k_product_mm_64x128<bits, 2><<<grid, 256, 0, stream>>>(codes, scales, in, out, K, N, batch);
    } else {
        const dim3 grid((N / 16 + 7) / 8, (batch + 127) / 128, 1);
        k_product_mm_128x128<bits, 2><<<grid, 256, 0, stream>>>(codes, scales, in, out, K, N, batch);
    }
}

static void product(const op_t & o, const float * in, float * out, int batch, bool wmma, cudaStream_t stream) {
    switch (o.bits) {
        case 1: product<1>(o, in, out, batch, wmma, stream); break;
        case 2: product<2>(o, in, out, batch, wmma, stream); break;
        case 3: product<3>(o, in, out, batch, wmma, stream); break;
        case 4: product<4>(o, in, out, batch, wmma, stream); break;
        case 5: product<5>(o, in, out, batch, wmma, stream); break;
        case 6: product<6>(o, in, out, batch, wmma, stream); break;
        default: GGML_ABORT("trellis hip: bad bit width %d", o.bits);
    }
}

static bool side_supported(int n, int kind, int block, int shards) {
    if (kind == 1) {
        return block <= 16384;
    }
    if (kind == 2) {
        return shards <= 65535 && n / shards <= 24576;
    }
    return true;
}

static bool supports(const ggml_tensor * op) {
    const op_t o(op);
    return o.K <= INT32_MAX - 255 && o.N <= INT32_MAX - 255 &&
           side_supported(o.N, o.rows_kind, o.rows_block, o.rows_shards) &&
           side_supported(o.K, o.cols_kind, o.cols_block, o.cols_shards);
}

static void mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const op_t o(dst);
    const int K = o.K, N = o.N;
    cudaStream_t stream = ctx.stream();

    const int64_t capacity = std::min<int64_t>(o.T, BATCH);
    const bool need_in_tmp  = o.cols_kind == 2 && ggml_trellis_hadfull_m(K / o.cols_shards) > 1;
    const bool need_out_tmp = o.rows_kind == 2 && ggml_trellis_hadfull_m(N / o.rows_shards) > 1;
    const size_t n_in  = (size_t) (need_in_tmp  ? 2 : 1) * capacity * K;
    const size_t n_out = (size_t) (need_out_tmp ? 2 : 1) * capacity * N;
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), n_in + n_out);
    float * const base = scratch.ptr;

    // matrix-core prefill only where the gfx12 WMMA tiles are validated
    const bool wmma = wmma_enabled();

    const bool fuse_in = o.cols_kind == 1 && o.cols_block >= 128 && o.cols_block <= BLOCK_LDS_MAX;
    const bool fuse_out = o.rows_kind == 1 && o.rows_block >= 128 && o.rows_block <= BLOCK_LDS_MAX && o.pout == nullptr;

    for (int64_t first = 0; first < o.T; first += BATCH) {
        const int batch = (int) std::min<int64_t>(o.T - first, BATCH);
        float * in      = base;
        float * in_tmp  = need_in_tmp ? base + capacity * K : nullptr;
        float * out     = base + n_in;
        float * out_tmp = need_out_tmp ? out + capacity * N : nullptr;

        float * din = in;  float * din_tmp = in_tmp;
        if (fuse_in) {
            const float sc = (float) std::pow((double) o.cols_block, -0.5);
            const size_t smem = o.cols_block > 128 ? (size_t) o.cols_block * sizeof(float) : 0;
            k_input_block_fast<<<dim3(K / o.cols_block, batch, 1), o.cols_block / 4, smem, stream>>>(
                K, o.cols_block, sc, (int) first, (int) o.x->ne[1], (int) o.x->ne[2], o.x->nb[1], o.x->nb[2], o.x->nb[3],
                (const char *) o.x->data, (const float *) o.pre->data,
                o.pin ? (const int32_t *) o.pin->data : nullptr, in);
        } else {
            const dim3 block(256, 1, 1);
            k_input<<<dim3((K + 255) / 256, batch, 1), block, 0, stream>>>(
                K, (int) first, (int) o.x->ne[1], (int) o.x->ne[2], o.x->nb[1], o.x->nb[2], o.x->nb[3],
                (const char *) o.x->data, (const float *) o.pre->data,
                o.pin ? (const int32_t *) o.pin->data : nullptr, in);
            side(din, din_tmp, K, batch, o.cols_kind, o.cols_block, o.cols_shards, nullptr, 0, false, stream);
        }

        product(o, din, out, batch, wmma, stream);

        if (fuse_out) {
            const float sc = (float) std::pow((double) o.rows_block, -0.5);
            const size_t smem = o.rows_block > 128 ? (size_t) o.rows_block * sizeof(float) : 0;
            k_output_block_fast<false><<<dim3(N / o.rows_block, batch, 1), o.rows_block / 4, smem, stream>>>(
                N, o.rows_block, sc, (int) first, out, (const float *) o.post->data, (float *) dst->data);
        } else {
            float * dout = out; float * dout_tmp = out_tmp;
            side(dout, dout_tmp, N, batch, o.rows_kind, o.rows_block, o.rows_shards, o.parts, o.nparts, true, stream);
            const dim3 block(256, 1, 1);
            k_output<<<dim3((N + 255) / 256, batch, 1), block, 0, stream>>>(
                N, (int) first, dout, (const float *) o.post->data,
                o.pout ? (const int32_t *) o.pout->data : nullptr, (float *) dst->data);
        }
    }
}

// Fuse MLP gate/up output Hadamard + SwiGLU + down input Hadamard into one kernel.
static int mlp_chain_size(const ggml_cgraph * cgraph, int i) {
    if (i + 3 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * a = cgraph->nodes[i];
    const ggml_tensor * b = cgraph->nodes[i + 1];
    const ggml_tensor * glu = cgraph->nodes[i + 2];
    const ggml_tensor * down = cgraph->nodes[i + 3];
    if (a->op != GGML_OP_TRELLIS || b->op != GGML_OP_TRELLIS || glu->op != GGML_OP_GLU || down->op != GGML_OP_TRELLIS) {
        return 0;
    }
    if (!ggml_cuda_trellis_supported(a) || !ggml_cuda_trellis_supported(b) || !ggml_cuda_trellis_supported(down) ||
        !supports(a) || !supports(b) || !supports(down)) {
        return 0;
    }
    if (ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(glu, 1) != 0 || glu->src[1] == nullptr ||
        glu->type != GGML_TYPE_F32 || !ggml_is_contiguous(glu)) {
        return 0;
    }
    const ggml_tensor * gate = glu->src[0];
    const ggml_tensor * up   = glu->src[1];
    if (!((gate == a && up == b) || (gate == b && up == a)) || down->src[1] != glu) {
        return 0;
    }
    const op_t og(gate), ou(up), od(down);
    const int blk = og.rows_block;
    if (og.T > BATCH || og.x != ou.x || og.K != ou.K ||
        og.rows_kind != 1 || ou.rows_kind != 1 || ou.rows_block != blk || og.pout != nullptr || ou.pout != nullptr ||
        od.cols_kind != 1 || od.cols_block != blk || od.pin != nullptr ||
        blk < 128 || blk > BLOCK_LDS_MAX || (blk & (blk - 1)) != 0 ||
        og.N != ou.N || od.K != og.N || og.N % blk != 0 || od.T != og.T) {
        return 0;
    }
    const ggml_op ops[4] = {GGML_OP_TRELLIS, GGML_OP_TRELLIS, GGML_OP_GLU, GGML_OP_TRELLIS};
    const int outputs[1] = {i + 3};
    if (!ggml_can_fuse_subgraph(cgraph, i, 4, ops, outputs, 1)) {
        return 0;
    }
    return 4;
}

static void mul_mat_mlp_chain(ggml_backend_cuda_context & ctx, ggml_tensor * const * nodes) {
    const ggml_tensor * glu = nodes[2];
    ggml_tensor * gate = glu->src[0];
    ggml_tensor * up   = glu->src[1];
    ggml_tensor * down = nodes[3];
    const op_t og(gate), ou(up), od(down);
    cudaStream_t stream = ctx.stream();

    const int batch = (int) og.T;
    const bool wmma = wmma_enabled();

    const bool need_g_tmp = og.cols_kind == 2 && ggml_trellis_hadfull_m(og.K / og.cols_shards) > 1;
    const bool need_u_tmp = ou.cols_kind == 2 && ggml_trellis_hadfull_m(ou.K / ou.cols_shards) > 1;
    const bool need_d_tmp = od.rows_kind == 2 && ggml_trellis_hadfull_m(od.N / od.rows_shards) > 1;
    const size_t n_in  = (size_t) ((need_g_tmp || need_u_tmp) ? 2 : 1) * batch * og.K;
    const size_t n_xd  = (size_t) batch * og.N;
    const size_t n_out = (size_t) (need_d_tmp ? 2 : 1) * batch * od.N;
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), n_in + n_xd + n_out);
    float * in      = scratch.ptr;
    float * in_tmp  = (need_g_tmp || need_u_tmp) ? in + (size_t) batch * og.K : nullptr;
    float * xd      = in + n_in;
    float * out_d   = xd + n_xd;
    float * out_tmp = need_d_tmp ? out_d + (size_t) batch * od.N : nullptr;

    auto run_input = [&](const op_t & o, float *& din) {
        din = in;
        float * din_tmp = in_tmp;
        if (o.cols_kind == 1 && o.cols_block >= 128 && o.cols_block <= BLOCK_LDS_MAX) {
            const float sc = (float) std::pow((double) o.cols_block, -0.5);
            const size_t smem = o.cols_block > 128 ? (size_t) o.cols_block * sizeof(float) : 0;
            k_input_block_fast<<<dim3(o.K / o.cols_block, batch, 1), o.cols_block / 4, smem, stream>>>(
                o.K, o.cols_block, sc, 0, (int) o.x->ne[1], (int) o.x->ne[2], o.x->nb[1], o.x->nb[2], o.x->nb[3],
                (const char *) o.x->data, (const float *) o.pre->data,
                o.pin ? (const int32_t *) o.pin->data : nullptr, in);
        } else {
            k_input<<<dim3((o.K + 255) / 256, batch, 1), dim3(256, 1, 1), 0, stream>>>(
                o.K, 0, (int) o.x->ne[1], (int) o.x->ne[2], o.x->nb[1], o.x->nb[2], o.x->nb[3],
                (const char *) o.x->data, (const float *) o.pre->data,
                o.pin ? (const int32_t *) o.pin->data : nullptr, in);
            side(din, din_tmp, o.K, batch, o.cols_kind, o.cols_block, o.cols_shards, nullptr, 0, false, stream);
        }
    };

    float * din = nullptr;
    run_input(og, din);
    product(og, din, (float *) gate->data, batch, wmma, stream);

    run_input(ou, din);
    product(ou, din, (float *) up->data, batch, wmma, stream);

    const int blk = og.rows_block;
    const float sc = (float) std::pow((double) blk, -0.5);
    const size_t smem = blk > 128 ? (size_t) 2 * blk * sizeof(float) : 0;
    k_swiglu_block_fast<<<dim3(og.N / blk, batch, 1), blk / 4, smem, stream>>>(
        (const float *) gate->data, (const float *) up->data, xd, og.N, blk, sc,
        (const float *) og.post->data, (const float *) ou.post->data, (const float *) od.pre->data);

    product(od, xd, out_d, batch, wmma, stream);

    if (od.rows_kind == 1 && od.rows_block >= 128 && od.rows_block <= BLOCK_LDS_MAX && od.pout == nullptr) {
        const float dsc = (float) std::pow((double) od.rows_block, -0.5);
        const size_t dsmem = od.rows_block > 128 ? (size_t) od.rows_block * sizeof(float) : 0;
        k_output_block_fast<false><<<dim3(od.N / od.rows_block, batch, 1), od.rows_block / 4, dsmem, stream>>>(
            od.N, od.rows_block, dsc, 0, out_d, (const float *) od.post->data, (float *) down->data);
    } else {
        float * dout = out_d;
        float * dout_tmp = out_tmp;
        side(dout, dout_tmp, od.N, batch, od.rows_kind, od.rows_block, od.rows_shards, od.parts, od.nparts, true, stream);
        k_output<<<dim3((od.N + 255) / 256, batch, 1), dim3(256, 1, 1), 0, stream>>>(
            od.N, 0, dout, (const float *) od.post->data,
            od.pout ? (const int32_t *) od.pout->data : nullptr, (float *) down->data);
    }
}

} // namespace trellis_hip_rdna4
