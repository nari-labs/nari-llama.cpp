#extension GL_EXT_shader_16bit_storage : require

layout(binding = 0) readonly buffer Codes { uint codes[]; };
layout(binding = 1) readonly buffer Scales { float16_t scales[]; };

layout(constant_id = 0) const uint BITS = 3;
const uint TILE_WORDS = 8 * BITS;
const uint TILE_BITS = 256 * BITS;

// The hash of a 16-bit state, state * 0x83dcd12d, as two 16 x 16-bit products, s * 0xd12d + (s * 0x83dc) << 16: on
// RDNA these run as full-rate 24-bit multiplies instead of one quarter-rate 32-bit multiply. The compiler folds the two
// products back into one 32-bit multiply when it sees the constants, so trellis_hash_init sets the halves and the shift
// from a value it cannot see (an always-zero bit of a push constant). Without the call the hash stays correct.
uint trellis_hash_lo = 0xd12du;
uint trellis_hash_hi = 0x83dcu;
uint trellis_hash_shift = 16u;

void trellis_hash_init(uint zero) {
    const uint c = 0x83dcd12du ^ zero;
    trellis_hash_lo = c & 65535u;
    trellis_hash_hi = c >> 16;
    trellis_hash_shift = 16u ^ zero;
}

uint trellis_hash(uint state) {
    const uint s = state & 65535u;
    return s * trellis_hash_lo + ((s * trellis_hash_hi) << trellis_hash_shift);
}

// The value of a state is the sum of these four 6-bit bytes minus 126.
uint trellis_bytes(uint state) {
    return (trellis_hash(state) >> 2) & 0x3f3f3f3fu;
}

// The same four 6-bit bytes, each times 4: no shift, and the dot products that sum them scale the result back.
uint trellis_bytes4(uint state) {
    return trellis_hash(state) & 0xfcfcfcfcu;
}

float trellis_grid(uint state) {
    // sum of the four bytes with shifts and adds instead of a multiply by 0x01010101 (a quarter-rate 32-bit integer
    // multiply on RDNA): every byte is below 64, so no sum carries out of its byte
    const uint b = trellis_bytes(state);
    const uint t = b + (b >> 16);
    return float(int((t + (t >> 8)) & 0xffu) - 126);
}

// Codes store cyclic 16-bit states in the exllamav3 16x16 tile order.
float trellis_value(uint tile, uint row, uint col) {
    const uint p = 8 * (4 * (row % 8) + (col % 8) / 2) + 4 * (row / 8) + 2 * (col / 8) + col % 2;
    const uint start = ((p + 1) * BITS + TILE_BITS - 16) % TILE_BITS;
    const uint word = start / 32;
    const uint offset = start % 32;
    const uint hi = codes[tile + word];
    uint state = hi >> (16 - min(offset, 16));
    if (offset > 16) {
        const uint lo = codes[tile + (word + 1) % TILE_WORDS];
        state = (hi << (offset - 16)) | (lo >> (48 - offset));
    }
    return trellis_grid(state);
}

uint trellis_state_start(uint p) {
    const uint end = (p + 1) * BITS;
    return end >= 16 ? end - 16 : end + TILE_BITS - 16;
}

// The four cyclic states starting at state p of a tile.
uvec4 trellis_states4(uint tile, uint p) {
    const uint start = trellis_state_start(p);
    const uint word = start / 32;
    const uint offset = start % 32;
    const uint hi = codes[tile + word];
    const uint lo = codes[tile + (word + 1 == TILE_WORDS ? 0 : word + 1)];
    // Callers start on a multiple of four states. For TRL1-6 this gives a nonzero bit offset.
    const uint window = (hi << offset) | (lo >> (32 - offset));
    uvec4 states;
    [[unroll]] for (uint j = 0; j < 4; ++j) {
        const uint bit = j * BITS;
        // Only the fourth TRL6 state needs the next two bits after the aligned word.
        states[j] = bit <= 16 ? window >> (16 - min(bit, 16)) : (window << 2) | ((lo >> ((30 - offset) & 31)) & 3);
    }
    return states;
}

// For TRL1-4 and p a multiple of eight, all eight states fit in two consecutive words.
void trellis_states8(uint tile, uint p, out uvec4 states_lo, out uvec4 states_hi) {
    const uint start = trellis_state_start(p);
    const uint word = start / 32;
    const uint offset = start % 32;
    const uint hi = codes[tile + word];
    const uint lo = codes[tile + (word + 1 == TILE_WORDS ? 0 : word + 1)];
    const uint window_lo = (hi << offset) | (lo >> (32 - offset));
    const uint window_hi = (window_lo << (4 * BITS)) | ((lo << offset) >> (32 - 4 * BITS));
    const uvec4 shifts = uvec4(16, 16 - BITS, 16 - 2 * BITS, 16 - 3 * BITS);
    states_lo = uvec4(window_lo) >> shifts;
    states_hi = uvec4(window_hi) >> shifts;
}

// The eight states from p (a multiple of eight) in two steps, so that a loop can read the words of the next tile
// before decoding this one: trellis_words8 gives the code words of the two halves (p and p + 4; for TRL1-4 the
// second pair is unused), the same for every tile, trellis_load8 reads them and trellis_states8_from decodes them.
uvec4 trellis_words8(uint p) {
    const uint a = trellis_state_start(p) / 32;
    const uint b = trellis_state_start(p + 4) / 32;
    return uvec4(a, a + 1 == TILE_WORDS ? 0 : a + 1, b, b + 1 == TILE_WORDS ? 0 : b + 1);
}

uvec4 trellis_load8(uint tile, uvec4 words) {
    if (BITS <= 4) {
        return uvec4(codes[tile + words.x], codes[tile + words.y], 0, 0);
    }
    return uvec4(codes[tile + words.x], codes[tile + words.y], codes[tile + words.z], codes[tile + words.w]);
}

uvec4 trellis_states4_from(uint hi, uint lo, uint offset) {
    const uint window = (hi << offset) | (lo >> (32 - offset));
    uvec4 states;
    [[unroll]] for (uint j = 0; j < 4; ++j) {
        const uint bit = j * BITS;
        states[j] = bit <= 16 ? window >> (16 - min(bit, 16)) : (window << 2) | ((lo >> ((30 - offset) & 31)) & 3);
    }
    return states;
}

void trellis_states8_from(uvec4 code, uint p, out uvec4 states_lo, out uvec4 states_hi) {
    if (BITS <= 4) {
        const uint offset = trellis_state_start(p) % 32;
        const uint window_lo = (code.x << offset) | (code.y >> (32 - offset));
        const uint window_hi = (window_lo << (4 * BITS)) | ((code.y << offset) >> (32 - 4 * BITS));
        const uvec4 shifts = uvec4(16, 16 - BITS, 16 - 2 * BITS, 16 - 3 * BITS);
        states_lo = uvec4(window_lo) >> shifts;
        states_hi = uvec4(window_hi) >> shifts;
    } else {
        states_lo = trellis_states4_from(code.x, code.y, trellis_state_start(p) % 32);
        states_hi = trellis_states4_from(code.z, code.w, trellis_state_start(p + 4) % 32);
    }
}

vec4 trellis_decode4(uint tile, uint p) {
    const uvec4 states = trellis_states4(tile, p);
    return vec4(trellis_grid(states.x), trellis_grid(states.y), trellis_grid(states.z), trellis_grid(states.w));
}
