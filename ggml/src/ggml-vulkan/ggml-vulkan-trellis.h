#pragma once

#include "ggml-vulkan-types.h"

struct vk_trellis_io {
    uint32_t k, n, first;
    uint32_t offsets[7];
    uint32_t strides[3];
    uint32_t ne1, ne2, use_gather;
    int32_t row0;
    uint32_t vocab;
};

// trellis_fused.comp: the input stage (with trellis_io.comp's offsets and strides) or the output stage, each with
// its block transform
struct vk_trellis_fused {
    uint32_t k, n, first;
    uint32_t offsets[7];
    uint32_t strides[3];
    uint32_t ne1, ne2, use_gather;
    uint32_t len, batch;
    float scale;
};

struct vk_trellis_product {
    uint32_t k, n, batch, scale_offset;
    // trellis_mm_big.comp: distance in uvec4 between the prepared activations of consecutive 256-token chunks
    // (workgroup z), when one launch multiplies two chunks
    uint32_t chunk_stride;
};

struct vk_trellis_transform {
    uint32_t width, len, run, m, shards, nparts;
    uint32_t parts[GGML_TRELLIS_MAX_PARTS];
    uint32_t step, inverse;
    float scale;
};

struct vk_trellis_out128 {
    uint32_t n, first, post_offset, dst_offset;
};

struct vk_trellis_prep_in {
    uint32_t k, batch, first, x_offset, x_stride, pre_offset, gather_offset, use_gather, scale_offset;
};

struct vk_trellis_chain_pc {
    uint32_t n, len;
    uint32_t gate, up, post_gate, post_up, pre_down, dst;
    float scale;
};

void ggml_vk_trellis_plan_chains(ggml_backend_vk_context * ctx, const ggml_cgraph * graph);
void ggml_vk_trellis_chain(ggml_backend_vk_context * ctx, vk_context & subctx, const ggml_tensor * glu);
size_t ggml_vk_trellis_scratch_stride(const vk_device & device, const ggml_tensor * op);
size_t ggml_vk_trellis_quant_size(const vk_device & device, const ggml_tensor * op);
bool ggml_vk_trellis_supported(const vk_device & device, const ggml_tensor * op);
uint32_t ggml_vk_trellis_profile_queries(const ggml_cgraph * graph);
void ggml_vk_trellis_print_profile(ggml_backend_vk_context * ctx);
void ggml_vk_trellis(ggml_backend_vk_context * ctx, vk_context & subctx, const ggml_tensor * op);
