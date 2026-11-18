"""Small known-value decoder fixture; set NARIQ_TEST_LIBRARY to enable native checks."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location('decode_reference', Path(__file__).resolve().parents[1] / 'decode_reference.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
gguf = m.gguf


class ReferenceTests(unittest.TestCase):
    def test_round_nearest_even(self):
        bits = np.array([0x3f808000, 0x3f818000, 0xbf808000, 0xbf818000], dtype=np.uint32)
        got = m.rounded_bf16(bits.view(np.float32)).view(np.uint16)
        np.testing.assert_array_equal(got, [0x3f80, 0x3f82, 0xbf80, 0xbf82])

    def test_invalid_gather(self):
        with self.assertRaises(ValueError):
            m.permutation(np.array([0, 0], dtype=np.int32), 2, 'gather')
        with self.assertRaises(ValueError):
            m.permutation(np.array([0, 2], dtype=np.int32), 2, 'gather')

    @unittest.skipUnless(os.environ.get('NARIQ_TEST_LIBRARY'), 'requires built ggml-base')
    def test_dense_bands_and_permutations(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / 'coded.gguf'
            w = gguf.GGUFWriter(src, 'qwen35')
            w.add_string('test.label', 'metadata preserved')
            w.add_array('test.array', [2, 7])
            w.add_tensor('plain.weight', np.array([0.25, -3], dtype=np.float16))
            for base, scale in [('token_embd.band00', 1), ('token_embd.band01', 2), ('blk.0.ffn_down', 3)]:
                w.add_array('trellis.desc.' + base, [0, 0, 1, 0, 0, 1, 0])
                w.add_tensor(base + '.weight', np.zeros((16, 4), dtype=np.uint8), raw_dtype=gguf.GGMLQuantizationType.TRL2)
                w.add_tensor(base + '.trellis_s16', np.array([scale], dtype=np.float16))
                w.add_tensor(base + '.trellis_pre', np.arange(1, 17, dtype=np.float32))
                w.add_tensor(base + '.trellis_post', np.arange(1, 17, dtype=np.float32))
                w.add_tensor(base + '.trellis_pin', np.arange(15, -1, -1, dtype=np.int32))
                w.add_tensor(base + '.trellis_pout', np.roll(np.arange(16, dtype=np.int32), 3))
            gather = np.roll(np.arange(32, dtype=np.int32), 7)
            w.add_tensor('token_embd.vocab_gather', gather)
            w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
            # All-zero circular codes decode to -126. Apply pre/post and permutations analytically.
            unit = -126 * np.outer(np.roll(np.arange(1, 17), 3), np.arange(16, 0, -1)).astype(np.float32)
            expected = np.concatenate([unit, 2 * unit])[gather]
            for dtype in ('f32', 'bf16'):
                dst = Path(tmp) / (dtype + '.gguf')
                record = m.convert(src, dst, os.environ['NARIQ_TEST_LIBRARY'], dtype, 2)
                r = gguf.GGUFReader(dst)
                tensors = {t.name: t for t in r.tensors}
                self.assertEqual(set(tensors), {'plain.weight', 'token_embd.weight', 'blk.0.ffn_down.weight'})
                self.assertFalse(any(k.startswith('trellis.') for k in r.fields))
                self.assertEqual(r.fields['test.array'].contents(), [2, 7])
                self.assertEqual(r.fields['test.label'].contents(), 'metadata preserved')
                np.testing.assert_array_equal(tensors['plain.weight'].data, [0.25, -3])
                for name, values in [('token_embd.weight', expected), ('blk.0.ffn_down.weight', 3 * unit)]:
                    t = tensors[name]
                    self.assertEqual(t.tensor_type, gguf.GGMLQuantizationType.BF16 if dtype == 'bf16' else gguf.GGMLQuantizationType.F32)
                    actual = gguf.quants.dequantize(t.data, t.tensor_type)
                    if dtype == 'bf16':
                        # Independent RNE on the exact integer fixture.
                        u = values.view(np.uint32)
                        values = ((u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000).view(np.float32)
                    np.testing.assert_array_equal(actual, values)
                self.assertEqual(r.fields['nariq.reference.source_sha256'].contents(), m.sha256(src))
                self.assertTrue(record['complete'])
                self.assertEqual(record['output_sha256'], m.sha256(dst))
                with self.assertRaises(ValueError):
                    m.convert(src, dst, os.environ['NARIQ_TEST_LIBRARY'])


if __name__ == '__main__':
    unittest.main()
