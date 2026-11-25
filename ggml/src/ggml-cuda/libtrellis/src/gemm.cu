// SPDX-License-Identifier: Apache-2.0
// Copyright Nari Labs
// libtrellis prefill GEMM (tq_gemm), any number of rows, with tile strides, fp32 / BF16 / fp16 inputs and outputs, and
// tq_gemv's pre, post and fused 128-point Hadamards. Whole-slice and larger rotations stay outside (tq_hadfull,
// tq_rht).
//
// 1. Prologue (one CTA per row): u = x * pre; with TQ_HAD128_IN, u = H_128 u / sqrt(128) per 128 inputs; u *= s16;
//    then fp16 in the m16n8k16 B-fragment order ([128-block][row group][k-tile][token][16 halves], tq_gemv's) with one
//    power-of-two scale per row: 2^e with max |u| 2^e in [2^13, 2^14).
// 2. GEMM: a CTA owns 128 outputs (8 output tiles, one per warp) x TT rows (64 or 128) over all of K (or one split of
//    it, item 4); no counters, nothing to zero. Per stage of CK k-tiles (4 or 8), cp.async stages the CTA's code
//    words and its rows' activations, S stages deep. Each warp decodes its tile once per k-tile into an A fragment of
//    4 v (exact in fp16, so no activation sums) and multiplies it with the B fragment of each of its TT / 8 row groups;
//    the next k-tile's decode is issued before this one's products. V never leaves registers.
// 3. Epilogue in the CTA, 32 rows at a time through shared memory: per row the optional output Hadamard, 1/4 (and
//    128^-1/2), the row's scale, post, and the store in the output type.
// 4. Split-K, for grids that would leave SMs idle (few column blocks at a few hundred rows: in_proj_a/b's 48 outputs, k/v_proj's 1,024).
//    Each split writes its rows' raw sums to fp32 partials in the workspace, and a second kernel adds the splits in order and runs the epilogue.
// Warps whose output tile is past N (N % 128 != 0) only help with the copies.
// Every tiling sums an output's k-tiles in the same order (one accumulator per (output, row, split), k-tile by
// k-tile), and the splits depend on the shape and the device only, so the result does not depend on the tiling, and
// two calls give the same bytes.
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdlib>
#include <atomic>
#include <cstdio>
#include <type_traits>

#include "common.cuh"

namespace {

size_t align256(size_t b) { return (b + 255) / 256 * 256; }

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void store_out(void* out, int type, int64_t idx, float v) {
    if (type == TQ_F32) static_cast<float*>(out)[idx] = v;
    else if (type == TQ_BF16) static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(v);
    else static_cast<__half*>(out)[idx] = __float2half_rn(v);
}

// ---- the fused path: prologue -------------------------------------------------------------------------------------

// the 48 KB default less room for the static wmax: rows of up to 12,160 inputs keep their values in shared memory
constexpr int PRO_BUF_BYTES = 48 * 1024 - 256;

// One CTA (8 warps) per row t. Each warp builds its 128-blocks (blk = warp, warp + 8, ...): pre, the optional Hadamard,
// s16; the row's max |u| gives e; then the warp stores its blocks as fp16 u 2^e. With `buf` the values wait in shared
// memory (each lane reads back only what it wrote); otherwise the second pass computes them again, the same way.
template <typename T>
__global__ void __launch_bounds__(256) prologue_k(const T* __restrict__ x, int64_t x_stride, int K,
                                                  const float* __restrict__ pre, int had,
                                                  const __half* __restrict__ s16, __half* __restrict__ xf,
                                                  int8_t* __restrict__ xf8, float* __restrict__ xs, int groups,
                                                  int buf) {
    extern __shared__ float rowbuf[];
    __shared__ float wmax[8];
    const int t = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32, nb = K / 128;
    const T* xr = x + (int64_t)t * x_stride;
    auto build = [&](int blk, float (&v)[4]) {
        const int k0 = 128 * blk;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int k = k0 + lane + 32 * i;
            v[i] = tq::to_f32<T>(xr[k]);
            if (pre != nullptr) v[i] *= __ldg(pre + k);
        }
        if (had) {
            tq::hadamard128(v, lane);
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] *= 0.08838834764831845f;
        }
        if (s16 != nullptr)
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] *= __half2float(s16[(k0 + lane + 32 * i) / 16]);
    };
    float mx = 0.f;
    for (int blk = warp; blk < nb; blk += 8) {
        float v[4];
        build(blk, v);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            mx = fmaxf(mx, fabsf(v[i]));
            if (buf) rowbuf[128 * blk + lane + 32 * i] = v[i];
        }
    }
    mx = tq::warp_max(mx);
    if (lane == 0) wmax[warp] = mx;
    __syncthreads();
    mx = wmax[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) mx = fmaxf(mx, wmax[w]);
    int x2 = 0;
    frexpf(mx > 0.f && mx <= 3.0e38f ? mx : 1.f, &x2);
    const int e = 14 - x2;
    if (threadIdx.x == 0) xs[t] = ldexpf(1.f, -e);
    __half* base = xf + (size_t)(t / 8) * 1024 + (t % 8) * 16;  // [c][group][k-tile][token][16]
    // INT8: 32 bytes per token per 32-column block, so a lane's two B words (k 4q..4q+3 and k 16+4q..16+4q+3)
    // are adjacent and the GEMM reads them with one LDS.64: [k32-block][group][token][32]
    int8_t* base8 = xf8 != nullptr ? xf8 + (size_t)(t / 8) * 1024 + (t % 8) * 32 : nullptr;
    const size_t cstep = (size_t)groups * 1024;
    for (int blk = warp; blk < nb; blk += 8) {
        float v[4];
        if (buf) {
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = rowbuf[128 * blk + lane + 32 * i];
        } else {
            build(blk, v);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int k = lane + 32 * i;
            if (base8 != nullptr) {
                // INT8: e - 7 puts max |u| at 2^6..2^7; byte (k%4) of word (k/4 within a k32 block) of its token
                int q = __float2int_rn(ldexpf(v[i], e - 7));
                q = q < -127 ? -127 : (q > 127 ? 127 : q);
                base8[blk * cstep + (k / 32) * 256 + ((k % 16) / 4) * 8 + ((k % 32) / 16) * 4 + (k % 4)] = static_cast<int8_t>(q);
            } else {
                base[blk * cstep + (k / 16) * 128 + tq::frag_pos(k % 16)] = __float2half_rn(ldexpf(v[i], e));
            }
        }
    }
}

// ---- the fused path: GEMM -----------------------------------------------------------------------------------------

constexpr int W5 = 8;          // warps = output tiles per CTA
constexpr int RED5 = 132;      // the epilogue's row stride in floats (128 + 4: conflict-free fragment stores)
constexpr int SMEM_SM = 102400;          // shared memory per SM to plan with (sm_86/89/120; sm_80/90 have more)
constexpr int SMEM_CTA = 101376 - 64;    // per CTA

// K bits, TT rows per CTA, CK k-tiles per stage, CPS CTAs per SM: the stage size and the stages that fit (at most 4)
template <int K, int TT, int CK, int CPS, bool I8 = false>
struct Tile5 {
    static constexpr int WORDS = 8 * K;
    static constexpr int CODE_W = W5 * CK * WORDS;                          // codes [tile][k-tile][8K]
    static constexpr int STAGE = CODE_W + TT / 8 * CK * (I8 ? 32 : 64);     // + activations (int8 halves the words)
    static constexpr int STAGE_BYTES = STAGE * 4;
    static constexpr int fit() {
        for (int s = 4; s >= 2; --s)
            if (CPS * ((s * STAGE_BYTES + 255) / 256 * 256 + 1024) <= SMEM_SM && s * STAGE_BYTES <= SMEM_CTA) return s;
        return 2;
    }
    static constexpr int S = fit();
    static constexpr int SMEM = S * STAGE_BYTES > 32 * RED5 * 4 ? S * STAGE_BYTES : 32 * RED5 * 4;
    static_assert(SMEM <= SMEM_CTA, "stages");
};

struct Pf {
    const uint32_t* codes;
    int kt, nt;
    int64_t sk, sn;  // tile strides
    const uint32_t* xf;
    const float* xs;
    int rows, groups;
    int kb, splits;  // k-tiles per split (a multiple of 8), splits (1: no partials)
    float* part;     // splits > 1: [splits][rows][N] fp32 sums, finished by reduce_k
    uint32_t mult;
    const float* post;
    int had;
    int i8;  // experiment: INT8 m16n8k32 path (TQ_I8=1); xf holds int8 activations
    void* out;
    int out_type;
    int64_t out_stride;
};

// out[t][o] from the scales: v (a row's 128 sums, lane l: outputs o0 + l + 32 m), the optional Hadamard, then
// v * 1/4 (* 128^-1/2) * post * the row's scale
__device__ __forceinline__ void finish_row(const Pf& a, float (&v)[4], const float (&scale)[4], int t, int o0, int nv) {
    const int lane = threadIdx.x % 32;
    if (a.had) tq::hadamard128(v, lane);
    const float sc = __ldg(a.xs + t);
#pragma unroll
    for (int m = 0; m < 4; ++m)
        if (lane + 32 * m < nv)
            store_out(a.out, a.out_type, (int64_t)t * a.out_stride + o0 + lane + 32 * m, v[m] * scale[m] * sc);
}

__device__ __forceinline__ void row_scales(const Pf& a, int o0, int nv, float (&scale)[4]) {
    constexpr float H128 = 0.08838834764831845f;  // 128^-1/2
    const int lane = threadIdx.x % 32;
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int o = lane + 32 * m;
        scale[m] = (a.had ? 0.25f * H128 : 0.25f) * (a.post != nullptr && o < nv ? __ldg(a.post + o0 + o) : 1.f);
    }
}

// split-K: one warp per (row, 128 outputs) sums the splits' partials in split order, then finishes the row
__global__ void __launch_bounds__(256) reduce_k(const Pf a) {
    const int lane = threadIdx.x % 32, cbs = (a.nt + W5 - 1) / W5;
    const int64_t item = (int64_t)blockIdx.x * 8 + threadIdx.x / 32;
    if (item >= (int64_t)a.rows * cbs) return;
    const int t = static_cast<int>(item / cbs), o0 = static_cast<int>(item % cbs) * 128;
    const int n = a.nt * 16, nv = min(128, n - o0);
    const size_t step = (size_t)a.rows * n;
    float v[4];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int o = lane + 32 * m;
        float sum = 0.f;
        if (o < nv) {
            const float* p = a.part + (size_t)t * n + o0 + o;
            for (int s = 0; s < a.splits; ++s) sum += __ldcg(p + s * step);
        }
        v[m] = sum;
    }
    float scale[4];
    row_scales(a, o0, nv, scale);
    finish_row(a, v, scale, t, o0, nv);
}

// D: the decode (0: offset decode, 1: FMA decode, see decode_frag), then the 1528 subtract
template <int K, int TT, int CK, int CPS, int D>
__global__ void __launch_bounds__(32 * W5, CPS) pf_k(const Pf a) {
    using T5 = Tile5<K, TT, CK, CPS>;
    constexpr int WORDS = T5::WORDS, CODE_W = T5::CODE_W, STAGE = T5::STAGE, S = T5::S;
    constexpr int GT = TT / 8;  // row groups per CTA, all of them in every warp
    extern __shared__ __align__(16) uint32_t sm[];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane >> 2, q = lane & 3;
    const int tb = blockIdx.x, cb = blockIdx.y, split = blockIdx.z, j0 = cb * W5, g0 = tb * GT;
    const int rg = min(GT, a.groups - g0);  // row groups present in this CTA
    const bool active = j0 + warp < a.nt;   // a warp past the last output tile only copies
    const int c0 = split * a.kb / CK, nch = min(a.kb, a.kt - split * a.kb) / CK;  // this split's stages
    // this thread's 16-byte copies per stage: code units (a tile's CK k-tiles are CK WORDS / 4 units, 2 K per k-tile),
    // activation units (CK 16 per row group: a row group's 8 k-tiles of a 128-block are 512 contiguous words)
    constexpr int UT = CK * WORDS / 4, UC = W5 * UT, NC = (UC + 32 * W5 - 1) / (32 * W5);
    constexpr int UA = CK * 16, NA = GT * UA / (32 * W5);
    static_assert(GT * UA % (32 * W5) == 0 && (CK == 4 || CK == 8), "stage copies");
    const uint32_t base = static_cast<uint32_t>(__cvta_generic_to_shared(sm));
    const uint32_t* csrc[NC];
    uint32_t cdst[NC];
    bool cok[NC];
#pragma unroll
    for (int i = 0; i < NC; ++i) {
        const int u = threadIdx.x + 32 * W5 * i, w = u / UT, uu = u % UT, d = uu / (2 * K), word = 4 * (uu % (2 * K));
        cok[i] = u < UC && j0 + w < a.nt;
        csrc[i] = a.codes + ((int64_t)d * a.sk + (int64_t)(j0 + w) * a.sn) * WORDS + word;
        cdst[i] = base + 4 * (w * CK * WORDS + 4 * uu);
    }
    const int64_t cstep = (int64_t)CK * a.sk * WORDS;  // words from one stage's codes to the next
    int asrc[NA];
    uint32_t adst[NA];
    bool aok[NA];
#pragma unroll
    for (int i = 0; i < NA; ++i) {
        const int u = threadIdx.x + 32 * W5 * i, r = u / UA, w = u % UA;
        aok[i] = r < rg;
        asrc[i] = (g0 + r) * 512 + 4 * w;
        adst[i] = base + 4 * (CODE_W + r * CK * 64 + 4 * w);
    }
    const size_t xstep = (size_t)a.groups * 512;  // words per 128-block
    auto issue = [&](int cl, int st) {  // the split's stage cl into buffer st
        const int c = c0 + cl;
        const uint32_t off = 4u * STAGE * st;
#pragma unroll
        for (int i = 0; i < NC; ++i)
            if (cok[i]) tq::cp16s(cdst[i] + off, csrc[i] + c * cstep);
        const uint32_t* xc = a.xf + (size_t)(c * CK / 8) * xstep + (c * CK % 8) * 64;
#pragma unroll
        for (int i = 0; i < NA; ++i)
            if (aok[i]) tq::cp16s(adst[i] + off, xc + asrc[i]);
    };
#pragma unroll 1
    for (int s = 0; s < S - 1; ++s) {
        if (s < nch) issue(s, s);
        tq::cp_commit();
    }
    const tq::Plan<K> plan(lane);
    float acc[GT][4];
#pragma unroll
    for (int r = 0; r < GT; ++r)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[r][e] = 0.f;
    // The K loop, once with every row group present (no predicates) and once for a partial last row tile
    auto kloop = [&](auto full) {
        constexpr bool FULL = decltype(full)::value;
        auto decode_tile = [&](const uint32_t* tw, int d, uint32_t (&af)[4]) {
            uint32_t w[2 * tq::Plan<K>::ANCH];
#pragma unroll
            for (int x = 0; x < tq::Plan<K>::ANCH; ++x) {
                w[2 * x] = tw[d * WORDS + plan.lo[x]];
                w[2 * x + 1] = tw[d * WORDS + plan.hi[x]];
            }
            tq::decode_exact<K, D>(w, plan, a.mult, af);
        };
#pragma unroll 1
        for (int c = 0, cur = 0, next = S - 1; c < nch; ++c) {
            tq::cp_wait<S - 2>();
            __syncthreads();  // stage c landed for every thread; stage c - 1's buffer is free
            if (c + S - 1 < nch) issue(c + S - 1, next);
            tq::cp_commit();
            const uint32_t* st = sm + cur * STAGE;
            cur = cur + 1 == S ? 0 : cur + 1;
            next = next + 1 == S ? 0 : next + 1;
            if (!active) continue;
            const uint32_t* tw = st + warp * CK * WORDS;
            const uint2* xb = reinterpret_cast<const uint2*>(st + CODE_W) + lane;
            uint32_t af[4];
            decode_tile(tw, 0, af);
#pragma unroll
            for (int d = 0; d < CK; ++d) {
                uint32_t an[4];
                if (d + 1 < CK) decode_tile(tw, d + 1, an);
#pragma unroll
                for (int r = 0; r < GT; ++r)
                    if (FULL || r < rg) {
                        const uint2 bb = xb[(r * CK + d) * 32];
                        mma16816(acc[r], af, bb.x, bb.y);
                    }
                if (d + 1 < CK) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) af[e] = an[e];
                }
            }
        }
    };
    if (rg == GT) kloop(std::true_type{});
    else kloop(std::false_type{});
    tq::cp_wait<0>();
    // epilogue: 32 rows at a time through shared memory (the stages are free once every warp is done); with splits,
    // each row's raw sums go to the partials instead (reduce_k finishes them)
    const int n = a.nt * 16, o0 = cb * 128, nv = min(128, n - o0);
    float scale[4];
    row_scales(a, o0, nv, scale);
    float* part = a.splits > 1 ? a.part + (size_t)split * a.rows * n + o0 : nullptr;
    float* red = reinterpret_cast<float*>(sm);
    __syncthreads();
#pragma unroll
    for (int p = 0; p < TT / 32; ++p) {
        // d0, d1: output 16 j + g, tokens 2q, 2q + 1 of the group; d2, d3: output 16 j + g + 8
#pragma unroll
        for (int r = 0; r < GT; ++r) {
            if (r / 4 != p) continue;
            const int tl = (r % 4) * 8 + 2 * q, ol = warp * 16 + g;
            red[tl * RED5 + ol] = acc[r][0];
            red[(tl + 1) * RED5 + ol] = acc[r][1];
            red[tl * RED5 + ol + 8] = acc[r][2];
            red[(tl + 1) * RED5 + ol + 8] = acc[r][3];
        }
        __syncthreads();
        for (int tl = warp; tl < 32; tl += W5) {
            const int t = tb * TT + p * 32 + tl;
            if (t >= a.rows) break;
            float v[4];
#pragma unroll
            for (int m = 0; m < 4; ++m) v[m] = red[tl * RED5 + lane + 32 * m];
            if (part != nullptr) {
#pragma unroll
                for (int m = 0; m < 4; ++m)
                    if (lane + 32 * m < nv) part[(size_t)t * n + lane + 32 * m] = v[m];
            } else {
                finish_row(a, v, scale, t, o0, nv);
            }
        }
        __syncthreads();
    }
}

// The tilings (all give the same bits): 64 rows x 8 k-tiles per stage, two CTAs per SM; 128 x 8, one; 128 x 4, two (at most 128 registers).
enum Tiling { T64X8 = 0, T128X8 = 1, T128X4 = 2 };

__device__ __forceinline__ uint32_t pack4(int a, int b, int c, int d) {
    return (uint32_t)(a & 0xFF) | ((uint32_t)(b & 0xFF) << 8) | ((uint32_t)(c & 0xFF) << 16) |
           ((uint32_t)(d & 0xFF) << 24);
}

__device__ __forceinline__ void mma16832(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// INT8 experiment (TQ_I8=1): m16n8k32 with the ring's 8 v's per lane repacked by warp shuffle.
// Weights are the exact integer grid v; activations are int8 with the row scale 2^(7-e) (see prologue_k).
template <int K, int TT, int CK, int CPS>
__global__ void __launch_bounds__(32 * W5, CPS) pf_k_i8(const Pf a) {
    using T5 = Tile5<K, TT, CK, CPS, true>;
    constexpr int WORDS = T5::WORDS, CODE_W = T5::CODE_W, STAGE = T5::STAGE, S = T5::S;
    constexpr int GT = TT / 8;
    extern __shared__ __align__(16) uint32_t sm[];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane >> 2, q = lane & 3;
    const int tb = blockIdx.x, cb = blockIdx.y, split = blockIdx.z, j0 = cb * W5, g0 = tb * GT;
    const int rg = min(GT, a.groups - g0);
    const bool active = j0 + warp < a.nt;
    const int c0 = split * a.kb / CK, nch = min(a.kb, a.kt - split * a.kb) / CK;
    constexpr int UT = CK * WORDS / 4, UC = W5 * UT, NC = (UC + 32 * W5 - 1) / (32 * W5);
    constexpr int UA = CK * 8, NA = GT * UA / (32 * W5);  // int8: 8 x 16 B per k-tile per row group
    static_assert(GT * UA % (32 * W5) == 0 && (CK == 4 || CK == 8), "stage copies");
    const uint32_t base = static_cast<uint32_t>(__cvta_generic_to_shared(sm));
    const uint32_t* csrc[NC];
    uint32_t cdst[NC];
    bool cok[NC];
#pragma unroll
    for (int i = 0; i < NC; ++i) {
        const int u = threadIdx.x + 32 * W5 * i, w = u / UT, uu = u % UT, d = uu / (2 * K), word = 4 * (uu % (2 * K));
        cok[i] = u < UC && j0 + w < a.nt;
        csrc[i] = a.codes + ((int64_t)d * a.sk + (int64_t)(j0 + w) * a.sn) * WORDS + word;
        cdst[i] = base + 4 * (w * CK * WORDS + 4 * uu);
    }
    const int64_t cstep = (int64_t)CK * a.sk * WORDS;
    int asrc[NA];
    uint32_t adst[NA];
    bool aok[NA];
#pragma unroll
    for (int i = 0; i < NA; ++i) {
        const int u = threadIdx.x + 32 * W5 * i, r = u / UA, w = u % UA;
        aok[i] = r < rg;
        asrc[i] = (g0 + r) * 256 + 4 * w;              // int8: 256 words (1 KiB) per group
        adst[i] = base + 4 * (CODE_W + r * CK * 32 + 4 * w);
    }
    const size_t xstep = (size_t)a.groups * 256;  // int8: 256 words (1 KiB) per group per 128-block
    auto issue = [&](int cl, int st) {
        const int c = c0 + cl;
        const uint32_t off = 4u * STAGE * st;
#pragma unroll
        for (int i = 0; i < NC; ++i)
            if (cok[i]) tq::cp16s(cdst[i] + off, csrc[i] + c * cstep);
        const uint32_t* xc = a.xf + (size_t)(c * CK / 8) * xstep + (c * CK % 8) * 32;
#pragma unroll
        for (int i = 0; i < NA; ++i)
            if (aok[i]) tq::cp16s(adst[i] + off, xc + asrc[i]);
    };
#pragma unroll 1
    for (int s = 0; s < S - 1; ++s) {
        if (s < nch) issue(s, s);
        tq::cp_commit();
    }
    const tq::Plan<K> plan(lane);
    int acc[GT][4];
#pragma unroll
    for (int r = 0; r < GT; ++r)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[r][e] = 0;
    // shuffle sources: cols 4q..4q+3 come from lanes with q' = 2q%4 and (2q+1)%4 (same g group);
    // q >= 2 reads that lane's second column pair (v2,v3 / v6,v7)
    const int src0 = (lane & ~3) | ((2 * q) & 3), src1 = (lane & ~3) | ((2 * q + 1) & 3);
    const bool usehi = q >= 2;
    // K <= 3: the four states this lane needs from one source span (5K + 16 <= 31 bits) fit one 32-bit window,
    // so both A registers are decoded in place and the shuffle disappears
    int dl[2], dh[2], ds[2];
    if constexpr (K <= 3) {
#pragma unroll
        for (int side = 0; side < 2; ++side) {
            const int src = (lane & ~3) | ((2 * q + side) & 3);
            const int end = ((8 * src + (q >= 2 ? 2 : 0) + 6) * K) % (256 * K);
            dh[side] = (end - 32 + 256 * K >> 5) % (8 * K);
            dl[side] = (dh[side] + 1) % (8 * K);
            ds[side] = 32 - (end & 31);
        }
    }
    auto decode_tile = [&](const uint32_t* tw, int d, uint32_t (&a0)[2]) {
        if constexpr (K <= 3) {
            int lo[4], hi[4];
#pragma unroll
            for (int side = 0; side < 2; ++side) {
                const uint32_t r = __funnelshift_rc(tw[d * WORDS + dl[side]], tw[d * WORDS + dh[side]], ds[side]);
                lo[2 * side] = tq::v_int(tq::low16<5 * K>(r), a.mult);
                lo[2 * side + 1] = tq::v_int(tq::low16<4 * K>(r), a.mult);
                hi[2 * side] = tq::v_int(tq::low16<K>(r), a.mult);
                hi[2 * side + 1] = tq::v_int(tq::low16<0>(r), a.mult);
            }
            a0[0] = pack4(lo[0], lo[1], lo[2], lo[3]);  // (g, 4q..4q+3)
            a0[1] = pack4(hi[0], hi[1], hi[2], hi[3]);  // (g+8, 4q..4q+3)
            return;
        }
        uint32_t w[2 * tq::Plan<K>::ANCH];
#pragma unroll
        for (int x = 0; x < tq::Plan<K>::ANCH; ++x) {
            w[2 * x] = tw[d * WORDS + plan.lo[x]];
            w[2 * x + 1] = tw[d * WORDS + plan.hi[x]];
        }
        int v[8];
        tq::decode_ints<K>(w, plan, a.mult, v);
        const uint32_t olo = pack4(v[0], v[1], v[4], v[5]);  // (g,2q),(g,2q+1),(g+8,2q),(g+8,2q+1)
        const uint32_t ohi = pack4(v[2], v[3], v[6], v[7]);  // cols +8
        // K > 3: two shuffles (odd q lanes carry the other half) + one byte permute per register
        const uint32_t x = __shfl_sync(0xffffffffu, (q & 1) ? ohi : olo, usehi ? src1 : src0);
        const uint32_t y = __shfl_sync(0xffffffffu, (q & 1) ? olo : ohi, usehi ? src0 : src1);
        const uint32_t even = usehi ? y : x, odd = usehi ? x : y;
        a0[0] = __byte_perm(even, odd, 0x5410);  // (g, 4q..4q+3)
        a0[1] = __byte_perm(even, odd, 0x7632);  // (g+8, 4q..4q+3)
    };
    auto kloop = [&](auto full) {
        constexpr bool FULL = decltype(full)::value;
#pragma unroll 1
        for (int c = 0, cur = 0, next = S - 1; c < nch; ++c) {
            tq::cp_wait<S - 2>();
            __syncthreads();
            if (c + S - 1 < nch) issue(c + S - 1, next);
            tq::cp_commit();
            const uint32_t* st = sm + cur * STAGE;
            cur = cur + 1 == S ? 0 : cur + 1;
            next = next + 1 == S ? 0 : next + 1;
            if (!active) continue;
            const uint32_t* tw = st + warp * CK * WORDS;
            const uint32_t* xb = st + CODE_W;
            uint32_t ap[4];
            decode_tile(tw, 0, *reinterpret_cast<uint32_t (*)[2]>(&ap[0]));
            decode_tile(tw, 1, *reinterpret_cast<uint32_t (*)[2]>(&ap[2]));
#pragma unroll
            for (int d = 0; d < CK; d += 2) {
                uint32_t an[4];
                if (d + 2 < CK) {
                    decode_tile(tw, d + 2, *reinterpret_cast<uint32_t (*)[2]>(&an[0]));
                    decode_tile(tw, d + 3, *reinterpret_cast<uint32_t (*)[2]>(&an[2]));
                }
#pragma unroll
                for (int r = 0; r < GT; ++r)
                    if (FULL || r < rg) {
                        const uint2 pair = reinterpret_cast<const uint2*>(xb)[((r * CK + d) / 2) * 32 + lane];
                        mma16832(acc[r], ap, pair.x, pair.y);
                    }
                if (d + 2 < CK) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) ap[e] = an[e];
                }
            }
        }
    };
    if (rg == GT) kloop(std::true_type{});
    else kloop(std::false_type{});
    tq::cp_wait<0>();
    const int n = a.nt * 16, o0 = cb * 128, nv = min(128, n - o0);
    float scale[4];
    row_scales(a, o0, nv, scale);
    __syncthreads();
#pragma unroll
    for (int p = 0; p < TT / 32; ++p) {
#pragma unroll
        for (int r = 0; r < GT; ++r) {
            if (r / 4 != p) continue;
            const int tl = (r % 4) * 8 + 2 * q, ol = warp * 16 + g;
            // int8: acc = sum(v * round(u * 2^(e-7))); 512 = 2^7 * 4 makes finish_row's 0.25 factor exact
            reinterpret_cast<float*>(sm)[tl * RED5 + ol] = (float)acc[r][0] * 512.f;
            reinterpret_cast<float*>(sm)[(tl + 1) * RED5 + ol] = (float)acc[r][1] * 512.f;
            reinterpret_cast<float*>(sm)[tl * RED5 + ol + 8] = (float)acc[r][2] * 512.f;
            reinterpret_cast<float*>(sm)[(tl + 1) * RED5 + ol + 8] = (float)acc[r][3] * 512.f;
        }
        __syncthreads();
        for (int tl = warp; tl < 32; tl += W5) {
            const int t = tb * TT + p * 32 + tl;
            if (t >= a.rows) break;
            float v[4];
#pragma unroll
            for (int m = 0; m < 4; ++m) v[m] = reinterpret_cast<const float*>(sm)[tl * RED5 + lane + 32 * m];
            finish_row(a, v, scale, t, o0, nv);
        }
        __syncthreads();
    }
}

template <int K, int TT, int CK, int CPS, int D>
cudaError_t launch_t(const Pf& a, cudaStream_t s) {
    using T5 = Tile5<K, TT, CK, CPS>;
    static std::atomic<bool> configured[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev >= 0 && dev < 64 && !configured[dev].load(std::memory_order_acquire)) {
        cudaError_t e = cudaFuncSetAttribute(pf_k<K, TT, CK, CPS, D>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             T5::SMEM);
        if (e == cudaSuccess)
            e = cudaFuncSetAttribute(pf_k<K, TT, CK, CPS, D>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        if (e != cudaSuccess) return e;
        configured[dev].store(true, std::memory_order_release);
    }
    const dim3 grid((a.rows + TT - 1) / TT, (a.nt + W5 - 1) / W5, a.splits);
    pf_k<K, TT, CK, CPS, D><<<grid, dim3(32 * W5), T5::SMEM, s>>>(a);
    if (a.splits > 1) {
        const int64_t items = (int64_t)a.rows * ((a.nt + W5 - 1) / W5);
        reduce_k<<<(unsigned)((items + 7) / 8), 256, 0, s>>>(a);
    }
    return cudaGetLastError();
}

template <int K, int D>
cudaError_t launch_d(const Pf& a, int tiling, cudaStream_t s) {
    if (tiling == T128X4) return launch_t<K, 128, 4, 2, D>(a, s);
    if (tiling == T128X8) return launch_t<K, 128, 8, 1, D>(a, s);
    return launch_t<K, 64, 8, 2, D>(a, s);
}

// always the offset decode (D = 0): the FMA decode is no faster in this kernel
template <int K>
cudaError_t launch_pf(const Pf& a, int tiling, cudaStream_t s) {
    return launch_d<K, 0>(a, tiling, s);
}

template <int K, int TT, int CK, int CPS>
cudaError_t launch_t_i8(const Pf& a, cudaStream_t s) {
    using T5 = Tile5<K, TT, CK, CPS, true>;
    static std::atomic<bool> configured[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev >= 0 && dev < 64 && !configured[dev].load(std::memory_order_acquire)) {
        cudaError_t e = cudaFuncSetAttribute(pf_k_i8<K, TT, CK, CPS>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, T5::SMEM);
        if (e == cudaSuccess)
            e = cudaFuncSetAttribute(pf_k_i8<K, TT, CK, CPS>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        if (e != cudaSuccess) return e;
        configured[dev].store(true, std::memory_order_release);
    }
    const int grid_t = (a.groups + TT / 8 - 1) / (TT / 8);
    dim3 grid(grid_t, (a.nt + W5 - 1) / W5, a.splits);
    pf_k_i8<K, TT, CK, CPS><<<grid, dim3(32 * W5), T5::SMEM, s>>>(a);
    return cudaGetLastError();
}

template <int K>
cudaError_t launch_pf_i8(const Pf& a, int tiling, cudaStream_t s) {
    if (tiling == T128X4) return launch_t_i8<K, 128, 4, 2>(a, s);
    if (tiling == T128X8) return launch_t_i8<K, 128, 8, 2>(a, s);  // CK=8 with 2 CTAs/SM (S=2): fewer barriers, more MMA per stage
    return launch_t_i8<K, 64, 8, 2>(a, s);
}

// The tiling whose most loaded SM finishes first, in units of one 128-row CTA alone on an SM (fitted on 188 SMs).
// 128 x 8: n CTAs per SM take n; 128 x 4, two per SM: a pair 1.8 and a single 1.03; 64 x 8, two per SM: a pair 1.1 and a single 0.6 (its CTAs hold half the rows).
int pick_tiling(int rows, int nt, int splits) {
    // debug override: TQ_TILING=0/1/2 forces T64X8/T128X8/T128X4
    if (const char * e = getenv("TQ_TILING")) {
        const int t = atoi(e);
        if (t >= 0 && t <= 2) return t;
    }
    const int cbs = (nt + W5 - 1) / W5, sms = tq::device_sms();
    const int c128 = cbs * ((rows + 127) / 128) * splits, c64 = cbs * ((rows + 63) / 64) * splits;
    auto load = [&](int ctas, float pair, float single) {
        const int n = (ctas + sms - 1) / sms;
        return (float)(n / 2) * pair + (float)(n % 2) * single;
    };
    const float t_c8 = (float)((c128 + sms - 1) / sms), t_x2 = load(c128, 1.8f, 1.03f), t_64 = load(c64, 1.1f, 0.6f);
    if (t_64 < t_x2 && t_64 < t_c8) return T64X8;
    return t_c8 < t_x2 ? T128X8 : T128X4;
}

// INT8 kernel: 128 x 8 with two CTAs per SM (measured best on the 36-SM part: 1303 vs 1263 for 128 x 4)
int pick_tiling_i8() {
    if (const char * e = getenv("TQ_TILING")) {
        const int t = atoi(e);
        if (t >= 0 && t <= 2) return t;
    }
    return T128X8;
}

// Split-K for grids that leave the GPU idle (few column blocks: in_proj_a/b's 48 outputs, k/v_proj's 1,024, at prefill
// row counts): splits of kb k-tiles (a multiple of 8, at least 32), about two 64-row CTAs per SM in all, at most 16.
// Each split sums its k-tiles in order into fresh accumulators; reduce_k adds the splits in order. The splits depend
// on (K, N, rows) and the device's SMs only, not on the tiling, so every tiling still gives the same bits.
constexpr int MAX_SPLITS = 16;

int split_count(int kt, int nt, int rows, int* kb) {
    const int64_t base = (int64_t)((nt + W5 - 1) / W5) * ((rows + 63) / 64);
    int s = (int)std::min<int64_t>(MAX_SPLITS, 2 * tq::device_sms() / base);
    s = std::min(s, kt / 32);
    if (s < 1) s = 1;
    *kb = ((kt + s - 1) / s + 7) / 8 * 8;
    return (kt + *kb - 1) / *kb;
}

struct FusedLayout {
    size_t xf, xs, part, total;
    int kb, splits;
};

FusedLayout fused_layout(int K, int N, int rows, bool i8) {
    const size_t groups = (size_t)(rows + 7) / 8;
    FusedLayout l;
    l.splits = split_count(K / 16, N / 16, rows, &l.kb);
    l.xf = 0;
    // keep the full fp16-sized xf region: splits > 1 still runs the fp16 kernel in i8 mode
    l.xs = align256((size_t)(K / 128) * groups * 2048);
    (void)i8;
    l.part = l.xs + align256((size_t)rows * sizeof(float));
    l.total = l.part + (l.splits > 1 ? align256((size_t)l.splits * rows * N * sizeof(float)) : 0);
    return l;
}

int fused(const tq_matrix* m, const void* x, tq_dtype x_type, int64_t x_stride, int rows, void* out, tq_dtype out_type,
          int64_t out_stride, const float* pre, const float* post, int flags, void* ws, size_t ws_bytes,
          cudaStream_t s) {
    if (tq::device_cc() < 80) return tq::fail(TQ_E_ARCH, "tq_gemm: needs sm_80 or newer (mma.sync m16n8k16)");
    if ((m->N / 16 + W5 - 1) / W5 > 65535) return tq::fail(TQ_E_ARG, "tq_gemm: N above 8,388,480");
    const char* i8env = std::getenv("TQ_I8");
    const bool want_i8 = i8env != nullptr && std::atoi(i8env) > 0;
    const FusedLayout l = fused_layout(m->K, m->N, rows, want_i8);
    if (ws == nullptr || ws_bytes < l.total || reinterpret_cast<uintptr_t>(ws) % 256)
        return tq::fail(TQ_E_WORKSPACE, "tq_gemm: workspace smaller than tq_gemm_workspace_bytes (or not 256-aligned)");
    char* w = static_cast<char*>(ws);
    __half* xf = reinterpret_cast<__half*>(w + l.xf);
    // splits > 1 goes through reduce_k (fp32 partials), which the int8 path does not cover
    int8_t* xf8 = want_i8 && l.splits == 1 ? reinterpret_cast<int8_t*>(w + l.xf) : nullptr;
    float* xs = reinterpret_cast<float*>(w + l.xs);
    const __half* s16 = static_cast<const __half*>(m->block_scale);
    const int K = m->K, groups = (rows + 7) / 8, hin = (flags & TQ_HAD128_IN) ? 1 : 0;
    const int buf = (size_t)K * 4 <= (size_t)PRO_BUF_BYTES ? 1 : 0;
    const size_t psm = buf ? (size_t)K * 4 : 0;
    switch (x_type) {
        case TQ_F32:
            prologue_k<float><<<rows, 256, psm, s>>>(static_cast<const float*>(x), x_stride, K, pre, hin, s16, xf, xf8,
                                                     xs, groups, buf);
            break;
        case TQ_BF16:
            prologue_k<__nv_bfloat16><<<rows, 256, psm, s>>>(static_cast<const __nv_bfloat16*>(x), x_stride, K, pre,
                                                             hin, s16, xf, xf8, xs, groups, buf);
            break;
        default:
            prologue_k<__half><<<rows, 256, psm, s>>>(static_cast<const __half*>(x), x_stride, K, pre, hin, s16, xf,
                                                      xf8, xs, groups, buf);
    }
    if (cudaError_t e = cudaGetLastError()) return tq::check(e, "tq_gemm prologue launch");
    Pf a;
    a.codes = static_cast<const uint32_t*>(m->codes);
    a.kt = K / 16, a.nt = m->N / 16, a.sk = m->tile_stride_k, a.sn = m->tile_stride_n;
    a.xf = reinterpret_cast<const uint32_t*>(w + l.xf), a.xs = xs;
    a.rows = rows, a.groups = groups, a.mult = m->mult;
    a.kb = l.kb, a.splits = l.splits, a.part = reinterpret_cast<float*>(w + l.part);
    a.post = post, a.had = (flags & TQ_HAD128_OUT) ? 1 : 0;
    a.i8 = xf8 != nullptr ? 1 : 0;
    a.out = out, a.out_type = out_type, a.out_stride = out_stride;
    const int tiling = a.i8 ? pick_tiling_i8() : pick_tiling(rows, a.nt, a.splits);
    cudaError_t e;
    if (a.i8) {
        switch (m->bits) {
            case 1: e = launch_pf_i8<1>(a, tiling, s); break;
            case 2: e = launch_pf_i8<2>(a, tiling, s); break;
            case 3: e = launch_pf_i8<3>(a, tiling, s); break;
            case 4: e = launch_pf_i8<4>(a, tiling, s); break;
            case 5: e = launch_pf_i8<5>(a, tiling, s); break;
            default: e = launch_pf_i8<6>(a, tiling, s); break;
        }
    } else {
        switch (m->bits) {
            case 1: e = launch_pf<1>(a, tiling, s); break;
            case 2: e = launch_pf<2>(a, tiling, s); break;
            case 3: e = launch_pf<3>(a, tiling, s); break;
            case 4: e = launch_pf<4>(a, tiling, s); break;
            case 5: e = launch_pf<5>(a, tiling, s); break;
            default: e = launch_pf<6>(a, tiling, s); break;
        }
    }
    return tq::check(e, "tq_gemm launch");
}

}  // namespace

extern "C" {
size_t tq_gemm_workspace_bytes(int K, int N, int rows) {
    if (K <= 0 || N <= 0 || K % 128 || N % 16 || rows < 1) return 0;
    return fused_layout(K, N, rows, false).total;  // size for the fp16 layout (the int8 path uses less)
}

int tq_gemm(const tq_matrix* m, const void* x, tq_dtype x_type, int64_t x_stride, int rows, void* out,
            tq_dtype out_type, int64_t out_stride, const float* pre, const float* post, int flags, void* ws,
            size_t ws_bytes, void* stream) {
    if (int e = tq::check_matrix(m, 128, "tq_gemm")) return e;
    if (rows < 1 || x == nullptr || out == nullptr || x_stride < m->K || out_stride < m->N || x_type < TQ_F32 ||
        x_type > TQ_F16 || out_type < TQ_F32 || out_type > TQ_F16)
        return tq::fail(TQ_E_ARG, "tq_gemm: rows >= 1, x and out with strides >= K and N");
    if ((flags & TQ_HAD128_OUT) && m->N % 128) return tq::fail(TQ_E_ARG, "tq_gemm: TQ_HAD128_OUT needs N % 128 == 0");
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    return fused(m, x, x_type, x_stride, rows, out, out_type, out_stride, pre, post, flags, ws, ws_bytes, s);
}
}
