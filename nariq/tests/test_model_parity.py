import array
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('compare_logits', Path(__file__).resolve().parents[1] / 'compare_logits.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ModelParityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def capture(self, name, values):
        prefix = self.root / name
        metadata = {'complete': True, 'prompt_ids': [1], 'evaluated_generation_ids': [0],
                    'vocab_size': len(values), 'logit_rows': 1, 'greedy_ids': [0]}
        Path(str(prefix) + '.json').write_text(json.dumps(metadata))
        with Path(str(prefix) + '.f32').open('wb') as f:
            array.array('f', values).tofile(f)
        return prefix

    def test_logit_offset_does_not_change_distribution(self):
        a = self.capture('2.5bit-cpu', [0, -1])
        b = self.capture('2.5bit-metal', [100, 99])
        result = module.compare(a, b, 1e-9, 1.0)
        self.assertTrue(result['passed'])
        self.assertLess(result['max_kl'], 1e-12)
        self.assertEqual(result['max_abs_logit_error'], 100)

    def test_full_distribution_detects_gap_with_same_top1(self):
        a = self.capture('cpu', [0, 0])
        b = self.capture('metal', [math.log(3), 0])
        result = module.compare(a, b, 1e-3, 1.0)
        self.assertEqual(result['top1_agreement'], 1)
        self.assertAlmostEqual(result['mean_kl'], 0.5 * math.log(4 / 3), places=7)
        self.assertFalse(result['passed'])

    def test_different_greedy_choices_fail(self):
        a = self.capture('cpu', [0, -1])
        b = self.capture('metal', [0, -1])
        metadata = Path(str(b) + '.json')
        content = json.loads(metadata.read_text())
        content['greedy_ids'] = [1]
        metadata.write_text(json.dumps(content))
        result = module.compare(a, b, 1e-3, 1)
        self.assertEqual(result['top1_agreement'], 1)
        self.assertFalse(result['greedy_ids_equal'])
        self.assertFalse(result['passed'])

    def test_truncated_and_nonfinite_captures_fail(self):
        a = self.capture('cpu', [0, -1])
        b = self.capture('metal', [0, float('nan')])
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            module.compare(a, b, 1e-3, 1)
        Path(str(b) + '.f32').write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'truncated'):
            module.compare(a, b, 1e-3, 1)


if __name__ == '__main__':
    unittest.main()
