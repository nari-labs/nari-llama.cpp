"""convert_hf_to_gguf.py on native trellis exports it cannot write; needs torch and transformers."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
                     'requires torch and transformers')
class ConvertTests(unittest.TestCase):
    def test_unsupported_architecture_is_refused(self):
        config = {
            'architectures': ['LlamaForCausalLM'], 'model_type': 'llama',
            'hidden_size': 256, 'intermediate_size': 512, 'num_attention_heads': 2, 'num_hidden_layers': 1,
            'num_key_value_heads': 2, 'rms_norm_eps': 1e-6, 'vocab_size': 32,
            'quantization_config': {'quant_method': 'native_trellis', 'modules': {}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'config.json').write_text(json.dumps(config))
            r = subprocess.run([sys.executable, str(ROOT / 'convert_hf_to_gguf.py'), tmp, '--outfile',
                                str(Path(tmp) / 'out.gguf')], capture_output=True, text=True, timeout=300)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('native_trellis exports are supported only for Qwen3.5', r.stderr)
            self.assertFalse((Path(tmp) / 'out.gguf').exists())


if __name__ == '__main__':
    unittest.main()
