#!/usr/bin/env python3
"""Build a dense numerical reference directly from a NariQ GGUF.

The decoder emits F32 in the original basis. BF16 rounds that result once,
nearest-even. Uncoded tensors retain their original representation.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-py"))
import gguf

TRL = {getattr(gguf.GGMLQuantizationType, f"TRL{i}") for i in range(1, 7)}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def permutation(values, size, name):
    values = np.asarray(values)
    if values.dtype != np.int32 or values.shape != (size,) or not np.array_equal(np.sort(values), np.arange(size)):
        raise ValueError(f"{name}: expected an I32 permutation of {size} entries")
    return values


def rounded_bf16(data):
    # gguf's converter implements round-to-nearest-even, including NaNs.
    return gguf.quants.quantize(np.ascontiguousarray(data), gguf.GGMLQuantizationType.BF16)


class Decoder:
    def __init__(self, reader, library):
        self.reader = reader
        self.tensors = {t.name: t for t in reader.tensors}
        self.lib = ctypes.CDLL(str(Path(library).resolve()))
        ptr = ctypes.c_void_p
        self.lib.ggml_trellis_desc_supported.argtypes = [ctypes.c_int64, ctypes.c_int64, ptr, ctypes.c_int]
        self.lib.ggml_trellis_desc_supported.restype = ctypes.c_bool
        self.lib.ggml_trellis_dequantize.argtypes = [ctypes.c_int, ptr, ctypes.c_int64, ctypes.c_int64,
                                                     ptr, ptr, ptr, ptr, ptr, ptr, ctypes.c_int, ptr]
        self.lib.ggml_trellis_dequantize.restype = ctypes.c_bool

    def inputs(self, tensor):
        if tensor.tensor_type not in TRL or len(tensor.shape) != 2 or not tensor.name.endswith('.weight'):
            raise ValueError(f"{tensor.name}: expected a coded weight matrix")
        k, n = map(int, tensor.shape)
        base = tensor.name[:-7]
        desc = np.asarray(self.reader.fields['trellis.desc.' + base].contents(), dtype=np.int32)
        if not self.lib.ggml_trellis_desc_supported(k, n, desc.ctypes.data, len(desc)):
            raise ValueError(f"{base}: unsupported descriptor")
        bits = int(tensor.tensor_type) - int(gguf.GGMLQuantizationType.TRL1) + 1
        if tensor.data.nbytes != k * n * bits // 8:
            raise ValueError(f"{base}: wrong code size")
        sides = []
        for suffix, size, dtype in [('s16', k // 16, np.float16), ('pre', k, np.float32),
                                    ('post', n, np.float32), ('pin', k, np.int32), ('pout', n, np.int32)]:
            name = base + '.trellis_' + suffix
            side = self.tensors.get(name)
            if side is None and suffix in ('pin', 'pout'):
                sides.append(None)
                continue
            if side is None or side.data.dtype != dtype or side.data.shape != (size,):
                raise ValueError(f"{name}: wrong shape or type")
            value = side.data
            if suffix in ('pin', 'pout'):
                permutation(value, size, name)
            elif not np.isfinite(value).all():
                raise ValueError(f"{name}: nonfinite scale")
            sides.append(value)
        return k, n, desc, sides

    def decode(self, tensor):
        k, n, desc, sides = self.inputs(tensor)
        output = np.empty((n, k), dtype=np.float32)
        args = [None if x is None else x.ctypes.data for x in sides]
        if not self.lib.ggml_trellis_dequantize(int(tensor.tensor_type), tensor.data.ctypes.data, k, n,
                                               *args, desc.ctypes.data, len(desc), output.ctypes.data):
            raise ValueError(f"{tensor.name}: decoding failed")
        if not np.isfinite(output).all():
            raise ValueError(f"{tensor.name}: nonfinite decoded weights")
        return output


def tensor_plan(decoder):
    tensors = decoder.tensors
    plans, skipped = [], set()
    for head in ('token_embd', 'output'):
        bands = sorted((t for t in tensors.values() if re.fullmatch(head + r'\.band\d+\.weight', t.name)),
                       key=lambda t: int(t.name.split('.band')[1].split('.')[0]))
        if not bands:
            continue
        if head + '.weight' in tensors:
            raise ValueError(f"{head}: both bands and a dense matrix")
        indices = [int(t.name.split('.band')[1].split('.')[0]) for t in bands]
        if indices != list(range(len(bands))):
            raise ValueError(f"{head}: missing band")
        k = int(bands[0].shape[0])
        if any(t.tensor_type not in TRL or len(t.shape) != 2 or int(t.shape[0]) != k for t in bands):
            raise ValueError(f"{head}: inconsistent bands")
        n = sum(int(t.shape[1]) for t in bands)
        gather = permutation(tensors[head + '.vocab_gather'].data, n, head + '.vocab_gather')
        plans.append((head + '.weight', (n, k), bands, np.argsort(gather)))
        skipped.update(t.name for t in bands)
        skipped.add(head + '.vocab_gather')
    for t in tensors.values():
        if '.trellis_' in t.name or t.name in skipped:
            continue
        if '.vocab_gather' in t.name or '.band' in t.name:
            raise ValueError(f"{t.name}: unsupported band layout")
        plans.append((t.name, tuple(map(int, t.shape[::-1])), [t], None))
    return plans


def convert(source, destination, library, outtype='bf16', threads=4):
    if outtype not in ('bf16', 'f32') or threads < 1:
        raise ValueError('Expected bf16 or f32 output and a positive worker count')
    source, destination = Path(source), Path(destination)
    partial = destination.with_name(destination.name + '.partial')
    manifest = destination.with_name(destination.name + '.json')
    if source.resolve() == destination.resolve() or any(p.exists() for p in (destination, partial, manifest)):
        raise ValueError('Destination, partial file and manifest must not already exist')
    reader = gguf.GGUFReader(source)
    if reader.endianess != gguf.GGUFEndian.LITTLE or sys.byteorder != 'little':
        raise ValueError('Only little-endian input and hosts are supported')
    decoder = Decoder(reader, library)
    coded = [t for t in reader.tensors if t.tensor_type in TRL]
    if not coded:
        raise ValueError('No NariQ weights found')
    for t in coded:
        decoder.inputs(t)  # Validate all buffers before entering native code or creating output.
    plans = tensor_plan(decoder)
    source_hash, library_hash = sha256(source), sha256(library)
    writer = gguf.GGUFWriter(partial, reader.fields['general.architecture'].contents())
    for name, field in reader.fields.items():
        if name.startswith(('GGUF.', 'trellis.', 'split.', 'nariq.reference.')) or name in ('general.architecture', 'general.file_type', 'general.quantization_version'):
            continue
        writer.add_key_value(name, field.contents(), field.types[0],
                             sub_type=field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None)
    for key, value in {'source_sha256': source_hash, 'decoder_sha256': library_hash,
                       'decoded_type': outtype, 'rounding': 'F32 decoder; BF16 round-nearest-even once'}.items():
        writer.add_string('nariq.reference.' + key, value)
    dtype = gguf.GGMLQuantizationType.BF16 if outtype == 'bf16' else gguf.GGMLQuantizationType.F32
    for name, shape, parts, inverse in plans:
        t = parts[0]
        decoded = inverse is not None or t.tensor_type in TRL
        writer.add_tensor_info(name, shape, np.dtype(np.float32),
                               int(np.prod(shape)) * (2 if outtype == 'bf16' else 4) if decoded else t.data.nbytes,
                               raw_dtype=dtype if decoded else t.tensor_type)
    record = {'source_sha256': source_hash, 'decoder_sha256': library_hash, 'decoded_type': outtype,
              'rounding': 'F32 decoder; BF16 round-nearest-even once', 'coded_tensors': len(coded),
              'uncoded_tensors': 'copied byte-for-byte', 'complete': False}
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        def materialize(plan):
            _, shape, parts, inverse = plan
            if inverse is not None:
                data = np.empty(shape, dtype=np.float32)
                offset = 0
                for part in parts:
                    decoded = decoder.decode(part)
                    end = offset + len(decoded)
                    data[inverse[offset:end]] = decoded
                    offset = end
                return rounded_bf16(data) if outtype == 'bf16' else data
            if parts[0].tensor_type in TRL:
                data = decoder.decode(parts[0])
                return rounded_bf16(data) if outtype == 'bf16' else data
            return parts[0].data

        with ThreadPoolExecutor(max_workers=threads) as pool:
            for start in range(0, len(plans), threads):
                batch = plans[start:start + threads]
                # Bound retained results; executor.map over the whole model would queue all weights.
                for offset, data in enumerate(pool.map(materialize, batch)):
                    writer.write_tensor_data(data)
                    del data
                    index = start + offset
                    print(f'{index + 1}/{len(plans)} {plans[index][0]}', flush=True)
        writer.close()
        record.update(complete=True, output_sha256=sha256(partial), output_bytes=partial.stat().st_size)
        # Exclusive hard-link creation avoids replacing an existing result.
        os.link(partial, destination)
        partial.unlink()
        with manifest.open('x') as stream:
            json.dump(record, stream, indent=2)
    except BaseException:
        writer.close()
        # A failed partial file is retained for diagnosis and cannot be mistaken for a final GGUF.
        raise
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--library', required=True, type=Path, help='libggml-base from this fork')
    parser.add_argument('--outtype', choices=('bf16', 'f32'), default='bf16')
    parser.add_argument('--threads', type=int, default=4, help='Maximum concurrent decoded matrices')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    convert(args.source, args.destination, args.library, args.outtype, args.threads)


if __name__ == '__main__':
    main()
