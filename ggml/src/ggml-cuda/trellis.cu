// trellis fork: the coded linear (GGML_OP_TRELLIS) on CUDA, through libtrellis (docs/llamacpp.md in the quantization
// repository): the optional input gather, the input side (fused into the GEMV when it is 128-point blockwise), tq_gemv
// for up to 64 columns or tq_gemm above, the output side and post, the optional output gather.
#include "trellis.cuh"

#ifdef GGML_CUDA_TRELLIS
#include "libtrellis.h"
#include "unary.cuh"

#include <cmath>
#include <set>
#include <string>

#define TRELLIS_CHECK(call)                                                                   \
    do {                                                                                      \
        const int rc_ = (call);                                                               \
        if (rc_ != 0) {                                                                       \
            GGML_ABORT("trellis: %s failed (%d): %s", #call, rc_, tq_last_error());           \
        }                                                                                     \
    } while (0)

// dst[t][j] = src[t][idx[j]] for j < n, t < rows
static __global__ void trellis_gather(const float * __restrict__ src, int64_t src_stride, const int32_t * __restrict__ idx,
                                      float * __restrict__ dst, int64_t dst_stride, int n) {
    const int64_t t = blockIdx.y;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) {
        dst[t * dst_stride + j] = src[t * src_stride + idx[j]];
    }
}

static void trellis_gather_launch(const float * src, int64_t src_stride, const int32_t * idx, float * dst,
                                  int64_t dst_stride, int n, int64_t rows, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = std::min<int>((n + threads - 1) / threads, 64);
    for (int64_t r0 = 0; r0 < rows; r0 += 65535) {
        const int64_t nr = std::min<int64_t>(rows - r0, 65535);
        trellis_gather<<<dim3(blocks, (unsigned) nr), threads, 0, stream>>>(src + r0 * src_stride, src_stride, idx,
                                                                            dst + r0 * dst_stride, dst_stride, n);
    }
    CUDA_CHECK(cudaGetLastError());
}

// up to 8 gathers of N x T products at once (an input group's output gathers): segment blockIdx.z,
// dst[t][j] = src[t][idx[j]] at stride n
struct trellis_gather_seg {
    const float * src;
    const int32_t * idx;
    float * dst;
    int n;
};
struct trellis_gather_set {
    trellis_gather_seg seg[8];
};

static __global__ void trellis_gather_multi(const trellis_gather_set set) {
    const trellis_gather_seg g = set.seg[blockIdx.z];
    const int64_t t = blockIdx.y;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < g.n; j += gridDim.x * blockDim.x) {
        g.dst[t * g.n + j] = g.src[t * g.n + g.idx[j]];
    }
}

static char * trellis_align256(char * p) {
    return (char *) (((uintptr_t) p + 255) & ~(uintptr_t) 255);
}

bool ggml_cuda_trellis_supported(const ggml_tensor * op) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    if (!ggml_is_trellis_type(w->type) || x->type != GGML_TYPE_F32 || x->nb[0] != sizeof(float)) {
        return false;
    }
    // columns evenly strided: T = ne1 ne2 ne3 rows of x at stride nb1
    return x->nb[2] == x->nb[1] * x->ne[1] && x->nb[3] == x->nb[2] * x->ne[2] && ggml_is_contiguous(op);
}

// the op's side description (docs/llamacpp.md: trellis.desc) and shapes
struct trellis_op {
    const ggml_tensor * w, * x, * s16, * pre, * post, * pin, * pout;
    int rows_kind, rows_block, rows_shards, cols_kind, cols_block, cols_shards, nparts;
    const int32_t * parts;
    int K, N, bits;
    int64_t T;

    explicit trellis_op(const ggml_tensor * dst) {
        w = dst->src[0], x = dst->src[1], s16 = dst->src[2], pre = dst->src[3], post = dst->src[4];
        pin = dst->src[5], pout = dst->src[6];
        const int32_t * d = (const int32_t *) dst->op_params;
        rows_kind = d[0], rows_block = d[1], rows_shards = d[2];
        cols_kind = d[3], cols_block = d[4], cols_shards = d[5], nparts = d[6], parts = d + 7;
        K = (int) w->ne[0], N = (int) w->ne[1];
        T = x->ne[1] * x->ne[2] * x->ne[3];
        bits = (int) w->type - (int) GGML_TYPE_TRL1 + 1;
    }
    tq_matrix matrix() const { return tq_matrix_rows(w->data, bits, K, N, s16->data); }
    bool in_fused() const { return cols_kind == 1 && cols_block == 128; }   // TQ_HAD128_IN
    bool out_fused() const { return rows_kind == 1 && rows_block == 128; }  // TQ_HAD128_OUT
    // the GEMV's post: the side's when it fuses (128-point) or there is none; else applied by the output side
    const float * gemv_post() const { return out_fused() || rows_kind == 0 ? (const float *) post->data : nullptr; }
};

// after the product y (N x T): the output side unless it fused, with post; then (gather) the output gather into dst
static void trellis_out(const trellis_op & o, float * y, float * dst, cudaStream_t stream, bool gather = true) {
    const float * b = (const float *) o.post->data;
    if (o.rows_kind == 1 && !o.out_fused()) {
        TRELLIS_CHECK(tq_rht(y, o.N, y, o.N, (int) o.T, o.N, o.rows_block, nullptr, b, stream));
    } else if (o.rows_kind == 2) {
        TRELLIS_CHECK(tq_hadfull(y, o.N, y, o.N, (int) o.T, o.N, o.parts, o.nparts, o.rows_shards, 1, nullptr, b,
                                 stream));
    }
    if (gather && o.pout != nullptr) {
        trellis_gather_launch(y, o.N, (const int32_t *) o.pout->data, dst, o.N, o.N, o.T, stream);
    }
}

static void trellis_product(ggml_backend_cuda_context & ctx, const trellis_op & o, const float * xin, int64_t xs,
                            const float * gpre, int flags, float * dst);

void ggml_cuda_op_trellis(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const trellis_op o(dst);
    const int K = o.K;
    const int64_t T = o.T;
    cudaStream_t stream = ctx.stream();
    const float * a = (const float *) o.pre->data;

    // input: gather, then the side unless it fuses into the GEMV
    const float * xin = (const float *) o.x->data;
    int64_t xs = o.x->nb[1] / sizeof(float);
    const bool in_sep = (o.cols_kind == 1 && !o.in_fused()) || o.cols_kind == 2;
    ggml_cuda_pool_alloc<float> xbuf(ctx.pool());
    if (o.pin != nullptr || in_sep) {
        xbuf.alloc((size_t) K * T);
    }
    if (o.pin != nullptr) {
        trellis_gather_launch(xin, xs, (const int32_t *) o.pin->data, xbuf.ptr, K, K, T, stream);
        xin = xbuf.ptr, xs = K;
    }
    int flags = 0;
    const float * gpre = nullptr;
    if (o.in_fused()) {
        flags |= TQ_HAD128_IN, gpre = a;
    } else if (o.cols_kind == 1) {
        TRELLIS_CHECK(tq_rht(xin, xs, xbuf.ptr, K, (int) T, K, o.cols_block, a, nullptr, stream));
        xin = xbuf.ptr, xs = K;
    } else if (o.cols_kind == 2) {
        TRELLIS_CHECK(tq_hadfull(xin, xs, xbuf.ptr, K, (int) T, K, nullptr, 0, o.cols_shards, 0, a, nullptr, stream));
        xin = xbuf.ptr, xs = K;
    } else {
        gpre = a;
    }

    trellis_product(ctx, o, xin, xs, gpre, flags, (float *) dst->data);
}

// the product of a prepared input xin (T rows of K at stride xs; gpre and flags & TQ_HAD128_IN as the input side leaves
// them), with the output side's post (and its 128-point Hadamard) fused where it can be; then the output side and the
// output gather into dst
static void trellis_product(ggml_backend_cuda_context & ctx, const trellis_op & o, const float * xin, int64_t xs,
                            const float * gpre, int flags, float * dst) {
    const int K = o.K, N = o.N;
    const int64_t T = o.T;
    cudaStream_t stream = ctx.stream();
    const tq_matrix m = o.matrix();
    float * y = dst;
    ggml_cuda_pool_alloc<float> ybuf(ctx.pool());
    if (o.pout != nullptr) {
        ybuf.alloc((size_t) N * T);
        y = ybuf.ptr;
    }
    if (o.out_fused()) {
        flags |= TQ_HAD128_OUT;
    }
    const float * gpost = o.gemv_post();
    if (T <= 64) {
        const size_t wsb = tq_gemv_workspace_bytes(K, N, (int) T);
        ggml_cuda_pool_alloc<char> wbuf(ctx.pool(), wsb + 256);
        char * ws = trellis_align256(wbuf.ptr);
        CUDA_CHECK(cudaMemsetAsync(ws, 0, TQ_GEMV_COUNTER_BYTES, stream));  // a pool buffer: the split counters start at 0
        TRELLIS_CHECK(tq_gemv(&m, xin, TQ_F32, xs, (int) T, y, TQ_F32, N, gpre, gpost, flags, ws, wsb, stream));
    } else {
        const size_t wsb = tq_gemm_workspace_bytes(K, N, (int) T);
        ggml_cuda_pool_alloc<char> wbuf(ctx.pool(), wsb + 256);
        char * ws = trellis_align256(wbuf.ptr);
        TRELLIS_CHECK(tq_gemm(&m, xin, TQ_F32, xs, (int) T, y, TQ_F32, N, gpre, gpost, flags, ws, wsb,
                              ctx.cublas_handle(), stream));
    }
    trellis_out(o, y, dst, stream);
}

static const int TRELLIS_GROUP_MAX = 8;            // tq_gemv_group's members
static const int64_t TRELLIS_GROUP_OUTPUTS = 2097152;  // and its outputs in all

// a node that can join an input group: a coded linear at up to 64 columns, no input gather, an input side the GEMV fuses
static bool trellis_groupable(const ggml_tensor * t) {
    if (t->op != GGML_OP_TRELLIS || (t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || !ggml_cuda_trellis_supported(t)) {
        return false;
    }
    const trellis_op o(t);
    return o.pin == nullptr && (o.cols_kind == 0 || o.in_fused()) && o.T <= 64;
}

// GGML_CUDA_TRELLIS_GROUP=0: every coded linear on its own, no input groups and no MLP chains (for A/B checks)
static bool trellis_grouping() {
    static const bool grouped = [] {
        const char * env = getenv("GGML_CUDA_TRELLIS_GROUP");
        return env == nullptr || atoi(env) != 0;
    }();
    return grouped;
}

int ggml_cuda_trellis_group_size(const ggml_cgraph * cgraph, int i) {
    const ggml_tensor * a = cgraph->nodes[i];
    if (!trellis_grouping() || !trellis_groupable(a)) {
        return 1;
    }
    int n = 1;
    int64_t outputs = a->src[0]->ne[1];
    while (n < TRELLIS_GROUP_MAX && i + n < cgraph->n_nodes) {
        const ggml_tensor * b = cgraph->nodes[i + n];
        if (!trellis_groupable(b) || b->src[1] != a->src[1] || b->src[0]->ne[0] != a->src[0]->ne[0] ||
            outputs + b->src[0]->ne[1] > TRELLIS_GROUP_OUTPUTS) {
            break;
        }
        outputs += b->src[0]->ne[1];
        ++n;
    }
    return n;
}

// the group's products; with sides, then each member's output side and its gather (else the raw products in each dst)
static void trellis_group(ggml_backend_cuda_context & ctx, ggml_tensor * const * nodes, int n, bool sides) {
    GGML_ASSERT(n >= 1 && n <= TRELLIS_GROUP_MAX);
    cudaStream_t stream = ctx.stream();
    const ggml_tensor * x = nodes[0]->src[1];
    const int64_t T = x->ne[1] * x->ne[2] * x->ne[3];
    const int64_t xs = x->nb[1] / sizeof(float);

    tq_matrix ms[TRELLIS_GROUP_MAX];
    void * outs[TRELLIS_GROUP_MAX];
    int64_t strides[TRELLIS_GROUP_MAX];
    const float * pre[TRELLIS_GROUP_MAX];
    const float * post[TRELLIS_GROUP_MAX];
    int flags[TRELLIS_GROUP_MAX];
    size_t y_total = 0;  // the members with an output gather write their product here first
    for (int i = 0; i < n; ++i) {
        const trellis_op o(nodes[i]);
        y_total += o.pout != nullptr ? (size_t) o.N * T : 0;
    }
    ggml_cuda_pool_alloc<float> ybuf(ctx.pool());
    if (y_total > 0) {
        ybuf.alloc(y_total);
    }
    float * yp = ybuf.ptr;
    for (int i = 0; i < n; ++i) {
        const trellis_op o(nodes[i]);
        GGML_ASSERT(o.x == x && o.T == T);
        ms[i] = o.matrix();
        pre[i] = (const float *) o.pre->data;
        post[i] = o.gemv_post();
        flags[i] = (o.in_fused() ? TQ_HAD128_IN : 0) | (o.out_fused() ? TQ_HAD128_OUT : 0);
        strides[i] = o.N;
        if (o.pout != nullptr) {
            GGML_ASSERT(sides);
            outs[i] = yp, yp += (size_t) o.N * T;
        } else {
            outs[i] = nodes[i]->data;
        }
    }
    const size_t wsb = tq_gemv_group_workspace_bytes(n, ms, (int) T);
    ggml_cuda_pool_alloc<char> wbuf(ctx.pool(), wsb + 256);
    char * ws = trellis_align256(wbuf.ptr);
    CUDA_CHECK(cudaMemsetAsync(ws, 0, TQ_GEMV_COUNTER_BYTES, stream));
    TRELLIS_CHECK(tq_gemv_group(n, ms, x->data, TQ_F32, xs, (int) T, outs, TQ_F32, strides, pre, post, flags, ws, wsb,
                                stream));
    if (!sides) {
        return;
    }
    // the output sides, then every member's output gather in one launch
    trellis_gather_set set = {};
    int nseg = 0, nmax = 0;
    for (int i = 0; i < n; ++i) {
        const trellis_op o(nodes[i]);
        trellis_out(o, (float *) outs[i], (float *) nodes[i]->data, stream, false);
        if (o.pout != nullptr) {
            set.seg[nseg++] = {(const float *) outs[i], (const int32_t *) o.pout->data, (float *) nodes[i]->data, o.N};
            nmax = std::max(nmax, o.N);
        }
    }
    if (nseg > 0) {
        const int threads = 256;
        const int blocks = std::min<int>((nmax + threads - 1) / threads, 64);
        trellis_gather_multi<<<dim3(blocks, (unsigned) T, nseg), threads, 0, stream>>>(set);
        CUDA_CHECK(cudaGetLastError());
    }
}

void ggml_cuda_op_trellis_group(ggml_backend_cuda_context & ctx, ggml_tensor * const * nodes, int n) {
    trellis_group(ctx, nodes, n, true);
}

// ---- the MLP chain: up and gate (an input group), swiglu(gate, up), down ------------------------------------------
// When up's and gate's output sides and down's input side are all rht<b> of one b, the two output rotations, the
// swiglu and down's input rotation run as one kernel over each block of b: x_d = H_b(pre_d * silu(g) * u) / sqrt(b)
// with g = post_g * H_b(gate) / sqrt(b), u likewise. The same float operations in the same order as tq_rht, ggml's
// swiglu and tq_rht one after another (the butterflies h = 1, 2, 4, ... as (a + b, a - b), then the scale, then post),
// so the same bytes.
// The butterflies h = 1 .. 16 E of a warp's 32 E entries (lane l, register e: entry l + 32 e) by shuffles and in
// registers, then h = 32 E .. b / 2 through shared memory, in order (libtrellis's rht_cta_k layout).
template <int E>
static __device__ __forceinline__ void trellis_warp_stages(float (&v)[E], int lane) {
#pragma unroll
    for (int h = 1; h < 32; h *= 2) {
        const bool hi = lane & h;
#pragma unroll
        for (int e = 0; e < E; ++e) {
            const float o = __shfl_xor_sync(0xffffffffu, v[e], h);
            v[e] = hi ? o - v[e] : v[e] + o;  // (a + b, a - b) of the pair (low a, high b)
        }
    }
#pragma unroll
    for (int g = 1; g < E; g *= 2) {
#pragma unroll
        for (int e = 0; e < E; ++e) {
            if (!(e & g)) {
                const float a = v[e], b = v[e + g];
                v[e] = a + b, v[e + g] = a - b;
            }
        }
    }
}

static __device__ __forceinline__ void trellis_smem_stages(float * s0, float * s1, int b, int h0) {
    for (int h = h0; h < b; h *= 2) {
        __syncthreads();
        const int sh = __ffs(h) - 1;
        for (int i = threadIdx.x; i < b / 2; i += blockDim.x) {
            const int lo = ((i >> sh) << (sh + 1)) | (i & (h - 1)), hi = lo + h;
            float a = s0[lo], c = s0[hi];
            s0[lo] = a + c, s0[hi] = a - c;
            if (s1 != nullptr) {
                a = s1[lo], c = s1[hi];
                s1[lo] = a + c, s1[hi] = a - c;
            }
        }
    }
    __syncthreads();
}

// one CTA of b / (32 E) warps per (block of b, column)
template <int E>
static __global__ void trellis_swiglu_rht(const float * __restrict__ g, const float * __restrict__ u,
                                          float * __restrict__ xd, int n, int b, float c,
                                          const float * __restrict__ post_g, const float * __restrict__ post_u,
                                          const float * __restrict__ pre_d) {
    extern __shared__ float s[];
    float * sg = s;
    float * su = s + b;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int64_t t = blockIdx.y;
    const int k0 = blockIdx.x * b;
    float vg[E], vu[E];
#pragma unroll
    for (int e = 0; e < E; ++e) {
        const int k = 32 * E * warp + lane + 32 * e;
        vg[e] = g[t * n + k0 + k];
        vu[e] = u[t * n + k0 + k];
    }
    trellis_warp_stages<E>(vg, lane);
    trellis_warp_stages<E>(vu, lane);
#pragma unroll
    for (int e = 0; e < E; ++e) {
        const int k = 32 * E * warp + lane + 32 * e;
        sg[k] = vg[e], su[k] = vu[e];
    }
    trellis_smem_stages(sg, su, b, 32 * E);
#pragma unroll
    for (int e = 0; e < E; ++e) {
        const int k = 32 * E * warp + lane + 32 * e;
        const float gv = sg[k] * c * post_g[k0 + k];
        const float uv = su[k] * c * post_u[k0 + k];
        vg[e] = (ggml_cuda_op_silu_single(gv) * uv) * pre_d[k0 + k];
    }
    trellis_warp_stages<E>(vg, lane);
    __syncthreads();  // every thread has read its sg entries above before any rewrites them
#pragma unroll
    for (int e = 0; e < E; ++e) {
        sg[32 * E * warp + lane + 32 * e] = vg[e];
    }
    trellis_smem_stages(sg, nullptr, b, 32 * E);
    for (int i = threadIdx.x; i < b; i += blockDim.x) {
        xd[t * n + k0 + i] = sg[i] * c;
    }
}

// GGML_CUDA_TRELLIS_DEBUG=1: why a candidate MLP chain was not fused (each reason once)
static int trellis_chain_reject(const char * why) {
    static const bool debug = [] {
        const char * env = getenv("GGML_CUDA_TRELLIS_DEBUG");
        return env != nullptr && atoi(env) != 0;
    }();
    if (debug) {
        static std::set<std::string> seen;
        if (seen.insert(why).second) {
            fprintf(stdout, "trellis: MLP chain not fused: %s\n", why);  // stdout: apart from the logs
            fflush(stdout);
        }
    }
    return 0;
}

int ggml_cuda_trellis_mlp_chain(const ggml_cgraph * cgraph, int i) {
    if (!trellis_grouping() || i + 3 >= cgraph->n_nodes || ggml_cuda_trellis_group_size(cgraph, i) != 2) {
        return 0;
    }
    const ggml_tensor * glu = cgraph->nodes[i + 2];
    const ggml_tensor * down = cgraph->nodes[i + 3];
    if (glu->op != GGML_OP_GLU) {
        return trellis_chain_reject(ggml_op_name(glu->op));  // the next node after an input group of two
    }
    if (ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(glu, 1) != 0 || glu->src[1] == nullptr ||
        glu->type != GGML_TYPE_F32 || !ggml_is_contiguous(glu)) {
        return trellis_chain_reject("not a split swiglu");
    }
    const ggml_tensor * gate = glu->src[0];
    const ggml_tensor * up = glu->src[1];
    const ggml_tensor * a = cgraph->nodes[i];
    const ggml_tensor * b = cgraph->nodes[i + 1];
    if (!((gate == a && up == b) || (gate == b && up == a))) {
        return trellis_chain_reject("the swiglu reads other tensors");
    }
    if (down->op != GGML_OP_TRELLIS || down->src[1] != glu || !ggml_cuda_trellis_supported(down)) {
        return trellis_chain_reject("the node after the swiglu is not its coded down");
    }
    const trellis_op og(gate), ou(up), od(down);
    const int blk = og.rows_block;
    if (og.rows_kind != 1 || ou.rows_kind != 1 || ou.rows_block != blk || og.pout != nullptr || ou.pout != nullptr ||
        od.cols_kind != 1 || od.cols_block != blk || od.pin != nullptr || blk < 256 || blk > 4096 ||
        og.N != ou.N || od.K != og.N || og.N % blk != 0 || od.T != og.T) {
        return trellis_chain_reject("sides or shapes");
    }
    const ggml_op ops[4] = {GGML_OP_TRELLIS, GGML_OP_TRELLIS, GGML_OP_GLU, GGML_OP_TRELLIS};
    const int outputs[1] = {i + 3};
    if (!ggml_can_fuse_subgraph(cgraph, i, 4, ops, outputs, 1)) {
        return trellis_chain_reject("up, gate or the swiglu is read elsewhere");
    }
    // No memory-range check: the chain reads and writes in the unfused order (x by the group; gate's and up's dst,
    // written raw, by the chain kernel into a pool buffer; that buffer by down, which writes its dst last), so down's
    // dst may share memory with x, gate or up (the allocator reuses them once read), as it may unfused.
    return 4;
}

void ggml_cuda_op_trellis_mlp_chain(ggml_backend_cuda_context & ctx, ggml_tensor * const * nodes) {
    const ggml_tensor * glu = nodes[2];
    ggml_tensor * down = nodes[3];
    const trellis_op og(glu->src[0]), ou(glu->src[1]), od(down);
    cudaStream_t stream = ctx.stream();
    const int blk = og.rows_block, n = og.N;
    const int64_t T = og.T;
    ggml_cuda_pool_alloc<float> xd(ctx.pool(), (size_t) n * T);
    trellis_group(ctx, nodes, 2, false);  // the raw products into gate's and up's dst
    const float c = (float) std::pow((double) blk, -0.5);
    constexpr int E = 4;  // 256-4096 entries: 2-32 warps
    trellis_swiglu_rht<E><<<dim3(n / blk, (unsigned) T), blk / E, 2 * blk * sizeof(float), stream>>>(
        (const float *) glu->src[0]->data, (const float *) glu->src[1]->data, xd.ptr, n, blk, c,
        (const float *) og.post->data, (const float *) ou.post->data, (const float *) od.pre->data);
    CUDA_CHECK(cudaGetLastError());
    trellis_product(ctx, od, xd.ptr, n, nullptr, 0, (float *) down->data);
}

#else

bool ggml_cuda_trellis_supported(const ggml_tensor * op) {
    GGML_UNUSED(op);
    return false;
}

void ggml_cuda_op_trellis(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("trellis: this build has no libtrellis (configure with -DGGML_CUDA_TRELLIS_DIR=<libtrellis folder>)");
}

int ggml_cuda_trellis_group_size(const ggml_cgraph * cgraph, int i) {
    GGML_UNUSED(cgraph);
    GGML_UNUSED(i);
    return 1;
}

void ggml_cuda_op_trellis_group(ggml_backend_cuda_context & ctx, ggml_tensor * const * nodes, int n) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(nodes);
    GGML_UNUSED(n);
    GGML_ABORT("trellis: this build has no libtrellis");
}

int ggml_cuda_trellis_mlp_chain(const ggml_cgraph * cgraph, int i) {
    GGML_UNUSED(cgraph);
    GGML_UNUSED(i);
    return 0;
}

void ggml_cuda_op_trellis_mlp_chain(ggml_backend_cuda_context & ctx, ggml_tensor * const * nodes) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(nodes);
    GGML_ABORT("trellis: this build has no libtrellis");
}

#endif
