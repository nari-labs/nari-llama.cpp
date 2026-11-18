#include "common.h"

kernel void kernel_trellis_input(
        constant ggml_metal_kargs_trellis & a [[buffer(0)]],
        device const char * x [[buffer(1)]],
        device const float * pre [[buffer(2)]],
        device const int * pin [[buffer(3)]],
        device float * dst [[buffer(4)]],
        uint2 i [[thread_position_in_grid]]) {
    if (i.x >= uint(a.K)) { return; }
    const ulong c = a.first + i.y;
    device const float * row = (device const float *)(x + (c % a.ne1)*a.nb1 + (c/a.ne1 % a.ne2)*a.nb2 + (c/(a.ne1*a.ne2))*a.nb3);
    dst[ulong(i.y)*a.K + i.x] = row[a.pin ? pin[i.x] : i.x]*pre[i.x];
}

// Map a logical whole-slice index back to the fused parts' stored order.
static ulong trellis_stored(constant ggml_metal_kargs_trellis_side & a, int l, int shard) {
    if (a.nparts == 0) { return ulong(shard)*a.len + l; }
    ulong start = 0;
    for (int p = 0; p < a.nparts; ++p) {
        const int size = a.parts[p]/a.shards;
        if (l < size) { return start + ulong(shard)*size + l; }
        l -= size;
        start += a.parts[p];
    }
    return start;
}

kernel void kernel_trellis_butterfly(
        constant ggml_metal_kargs_trellis_side & a [[buffer(0)]],
        device float * v [[buffer(1)]],
        uint3 i [[thread_position_in_grid]]) {
    if (i.x >= uint(a.len/2)) { return; }
    const int l = (i.x/a.h)*(2*a.h) + i.x%a.h;
    const ulong j = ulong(i.z)*a.n + trellis_stored(a, l, i.y);
    const ulong k = ulong(i.z)*a.n + trellis_stored(a, l + a.h, i.y);
    const float x = v[j], y = v[k];
    v[j] = (x + y)*a.scale;
    v[k] = (x - y)*a.scale;
}

static int trellis_chi(int d, int q) {
    d = (d % q + q) % q;
    if (d == 0) { return 0; }
    for (int i = 1; i < q; ++i) {
        if (i*i % q == d) { return 1; }
    }
    return -1;
}

kernel void kernel_trellis_cross(
        constant ggml_metal_kargs_trellis_side & a [[buffer(0)]],
        device const float * src [[buffer(1)]],
        device float * dst [[buffer(2)]],
        uint3 i [[thread_position_in_grid]]) {
    if (i.x >= uint(a.len)) { return; }
    const int row = i.x/a.run, j = i.x%a.run;
    float sum = 0;
    for (int b = 0; b < a.m; ++b) {
        const int r = a.inverse ? b : row, c = a.inverse ? row : b;
        const int s = (r == c) ? 1 : r == 0 ? 1 : c == 0 ? -1 : trellis_chi(c - r, a.m - 1);
        const float x = src[ulong(i.z)*a.n + trellis_stored(a, b*a.run + j, i.y)];
        sum += s > 0 ? x : -x;
    }
    dst[ulong(i.z)*a.n + trellis_stored(a, i.x, i.y)] = sum*a.scale;
}

// Sum four six-bit byte fields without carries between bytes.
static float trellis_value(uint state) {
    const uint h = state*0x83DCD12Du;
    return float(int((((h >> 2)&0x3f3f3f3fu)*0x01010101u) >> 24) - 126);
}

template<int bits, int columns>
kernel void kernel_trellis_product(
        constant ggml_metal_kargs_trellis & a [[buffer(0)]],
        device const uint * codes [[buffer(1)]],
        device const half * scales [[buffer(2)]],
        device const float * x [[buffer(3)]],
        device float * y [[buffer(4)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint sg [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]) {
    const uint row = 4*group.x + sg;
    const uint column = columns*group.y;
    const uint r = row%16, c = lane%16;
    const uint p = 8*(4*(r%8) + (c%8)/2) + 4*(r/8) + 2*(c/8) + c%2;
    constexpr uint nw = 8*bits, total = 256*bits;
    const uint start = ((p + 1)*bits + total - 16)%total;
    const uint word = start/32, offset = start%32;
    float sum[columns] = {};
    for (uint k = lane; k < uint(a.K); k += 32) {
        device const uint * tile = codes + (ulong(row/16)*(a.K/16) + k/16)*nw;
        const uint hi = tile[word], lo = tile[(word + 1)%nw];
        const uint state = (offset <= 16 ? hi >> (16 - offset) : (hi << (offset - 16)) | (lo >> (48 - offset))) & 65535;
        const float v = trellis_value(state);
        const float w = v*float(scales[k/16]);
        for (int j = 0; j < columns; ++j) {
            if (column + j < uint(a.batch)) { sum[j] += w*x[ulong(column + j)*a.K + k]; }
        }
    }
    for (int j = 0; j < columns; ++j) {
        const float value = simd_sum(sum[j]);
        if (lane == 0 && column + j < uint(a.batch)) { y[ulong(column + j)*a.N + row] = value; }
    }
}

#define TRELLIS_PRODUCT(B, C) \
template [[host_name("kernel_trellis_product_" #B "_" #C)]] \
kernel void kernel_trellis_product<B, C>(constant ggml_metal_kargs_trellis &, device const uint *, device const half *, device const float *, device float *, uint2, uint, uint);
TRELLIS_PRODUCT(1, 1)
TRELLIS_PRODUCT(2, 1)
TRELLIS_PRODUCT(3, 1)
TRELLIS_PRODUCT(4, 1)
TRELLIS_PRODUCT(5, 1)
TRELLIS_PRODUCT(6, 1)
TRELLIS_PRODUCT(1, 4)
TRELLIS_PRODUCT(2, 4)
TRELLIS_PRODUCT(3, 4)
TRELLIS_PRODUCT(4, 4)
TRELLIS_PRODUCT(5, 4)
TRELLIS_PRODUCT(6, 4)
#undef TRELLIS_PRODUCT

template<int bits>
static void trellis_decode8(device const uint * tile, uint lane, thread float * values) {
    constexpr uint nw = 8*bits, total = 256*bits;
    const uint start = ((8*lane + 1)*bits + total - 16)%total;
    const uint word = start/32, offset = start%32;
    const uint hi = tile[word], lo = tile[(word + 1)%nw];
    // Eight adjacent states share a 32-bit window at one or two bits per weight.
    if (bits <= 2) {
        const uint window = (hi << offset) | (lo >> (32 - offset));
        #pragma unroll
        for (uint j = 0; j < 8; ++j) {
            const uint state = (window >> (16 - j*bits)) & 65535;
            values[j] = trellis_value(state);
        }
    } else {
        const uint start4 = ((8*lane + 5)*bits + total - 16)%total;
        const uint word4 = start4/32, offset4 = start4%32;
        const uint hi4 = tile[word4];
        const uint lo4 = tile[(word4 + 1)%nw];
        #pragma unroll
        for (uint j = 0; j < 8; ++j) {
            const uint a = j < 4 ? hi : hi4, b = j < 4 ? lo : lo4;
            const uint o = (j < 4 ? offset : offset4) + (j%4)*bits;
            const uint state = (o <= 16 ? a >> (16 - o) : o < 32 ? (a << (o - 16)) | (b >> (48 - o)) : b >> (48 - o)) & 65535;
            values[j] = trellis_value(state);
        }
    }
}

template<int bits>
kernel void kernel_trellis_product_mv(
        constant ggml_metal_kargs_trellis & a [[buffer(0)]],
        device const uint * codes [[buffer(1)]],
        device const half * scales [[buffer(2)]],
        device const float * x [[buffer(3)]],
        device float * y [[buffer(4)]],
        uint group [[threadgroup_position_in_grid]],
        uint sg [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]) {
    // Four rows per SIMD group, with four independent K partitions.
    const uint tile_row = group;
    const uint c = 2*(lane%4), r = 16*tile_row + 2*sg + (lane%8)/4;
    float4 sum0 = 0, sum1 = 0;
    for (uint k = lane/8; k < uint(a.K/16); k += 4) {
        device const uint * tile = codes + (ulong(tile_row)*(a.K/16) + k)*(8*bits);
        float v[8];
        trellis_decode8<bits>(tile, 8*sg + lane%8, v);
        const float s = float(scales[k]);
        const float4 xv = float4(x[16*k + c], x[16*k + c + 1], x[16*k + c + 8], x[16*k + c + 9]);
        sum0 += (float4(v[0], v[1], v[2], v[3])*s)*xv;
        sum1 += (float4(v[4], v[5], v[6], v[7])*s)*xv;
    }
    sum0 += simd_shuffle_down(sum0, 16);
    sum1 += simd_shuffle_down(sum1, 16);
    sum0 += simd_shuffle_down(sum0, 8);
    sum1 += simd_shuffle_down(sum1, 8);
    sum0 += simd_shuffle_down(sum0, 2);
    sum1 += simd_shuffle_down(sum1, 2);
    sum0 += simd_shuffle_down(sum0, 1);
    sum1 += simd_shuffle_down(sum1, 1);
    if (lane < 8 && lane%4 == 0) {
        y[r] = sum0.x + sum0.y + sum0.z + sum0.w;
        y[r + 8] = sum1.x + sum1.y + sum1.z + sum1.w;
    }
}

#define TRELLIS_MV(B) \
template [[host_name("kernel_trellis_product_mv_" #B)]] \
kernel void kernel_trellis_product_mv<B>(constant ggml_metal_kargs_trellis &, device const uint *, device const half *, device const float *, device float *, uint, uint, uint);
TRELLIS_MV(1)
TRELLIS_MV(2)
TRELLIS_MV(3)
TRELLIS_MV(4)
TRELLIS_MV(5)
TRELLIS_MV(6)
#undef TRELLIS_MV

kernel void kernel_trellis_output(
        constant ggml_metal_kargs_trellis & a [[buffer(0)]],
        device const float * y [[buffer(1)]],
        device const float * post [[buffer(2)]],
        device const int * pout [[buffer(3)]],
        device float * dst [[buffer(4)]],
        uint2 i [[thread_position_in_grid]]) {
    if (i.x >= uint(a.N)) { return; }
    const uint j = a.pout ? pout[i.x] : i.x;
    dst[ulong(a.first + i.y)*a.N + i.x] = y[ulong(i.y)*a.N + j]*post[j];
}

template<int bits>
kernel void kernel_trellis_product_mm(
        constant ggml_metal_kargs_trellis & a [[buffer(0)]],
        device const uint * codes [[buffer(1)]],
        device const half * scales [[buffer(2)]],
        device const float * x [[buffer(3)]],
        device float * y [[buffer(4)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint BK = 64;
    threadgroup float sw[16*BK], sx[BK*16], sy[256];
    const uint row0 = 16*group.x, token0 = 16*group.y;
    const uint rb = 8*(sg%2), tb = 8*(sg/2);
    simdgroup_float8x8 acc(0.0f);
    constexpr uint nw = 8*bits;
    const uint lane = tid%32;
    for (uint first = 0; first < uint(a.K); first += BK) {
        const uint tile_k = first/16 + sg;
        float v[8] = {};
        if (tile_k < uint(a.K/16)) {
            device const uint * tile = codes + (ulong(group.x)*(a.K/16) + tile_k)*nw;
            trellis_decode8<bits>(tile, lane, v);
        }
        const float s = tile_k < uint(a.K/16) ? float(scales[tile_k]) : 0;
        #pragma unroll
        for (uint j = 0; j < 8; ++j) {
            const uint r = lane/4 + 8*(j/4);
            const uint c = 2*(lane%4) + j%2 + 8*(j/2%2);
            sw[r*BK + 16*sg + c] = v[j]*s;
        }
        for (uint p = tid; p < BK*16; p += 128) {
            const uint k = first + p/16, token = token0 + p%16;
            sx[p] = token < uint(a.batch) && k < uint(a.K) ? x[ulong(token)*a.K + k] : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint k = 0; k < BK; k += 8) {
            simdgroup_float8x8 w, xv;
            simdgroup_load(w, sw + rb*BK + k, BK);
            simdgroup_load(xv, sx + k*16 + tb, 16);
            simdgroup_multiply_accumulate(acc, w, xv, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc, sy + rb*16 + tb, 16);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < 256; p += 128) {
        const uint token = token0 + p%16;
        if (token < uint(a.batch)) { y[ulong(token)*a.N + row0 + p/16] = sy[p]; }
    }
}

#define TRELLIS_MM(B) \
template [[host_name("kernel_trellis_product_mm_" #B)]] \
kernel void kernel_trellis_product_mm<B>(constant ggml_metal_kargs_trellis &, device const uint *, device const half *, device const float *, device float *, uint2, uint, uint);
TRELLIS_MM(1)
TRELLIS_MM(2)
TRELLIS_MM(3)
TRELLIS_MM(4)
TRELLIS_MM(5)
TRELLIS_MM(6)
#undef TRELLIS_MM

kernel void kernel_trellis_block(
        constant ggml_metal_kargs_trellis_side & a [[buffer(0)]],
        device float * v [[buffer(1)]],
        threadgroup float * values [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint2 threads [[threads_per_threadgroup]]) {
    const ulong base = ulong(group.y)*a.n + ulong(group.x)*a.len;
    for (uint i = tid; i < uint(a.len); i += threads.x) { values[i] = v[base + i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h = 1; h < uint(a.len); h *= 2) {
        for (uint j = tid; j < uint(a.len/2); j += threads.x) {
            const uint i = (j/h)*(2*h) + j%h;
            const float x = values[i], y = values[i + h];
            values[i] = x + y;
            values[i + h] = x - y;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = tid; i < uint(a.len); i += threads.x) { v[base + i] = values[i]*a.scale; }
}
