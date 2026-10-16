"""CPU tests only: no network, model imports, real weights or CUDA."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import reproduce


class ConfigureTest(unittest.TestCase):
    def test_configure_keeps_native_normalization_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights, dino = root / 'weights', root / 'dino'
            names = {
                'TRELLIS-image-large': ['ss_dec_conv3d_16l8_fp16'],
                'TRELLIS.2-4B': ['ss_flow_img_dit_1_3B_64_bf16', 'shape_dec_next_dc_f16c32_fp16',
                                  'slat_flow_img2shape_dit_1_3B_512_bf16', 'slat_flow_img2shape_dit_1_3B_1024_bf16',
                                  'tex_dec_next_dc_f16c32_fp16', 'slat_flow_imgshape2tex_dit_1_3B_512_bf16',
                                  'slat_flow_imgshape2tex_dit_1_3B_1024_bf16']}
            for folder, stems in names.items():
                directory = weights / folder / 'ckpts'
                directory.mkdir(parents=True)
                for stem in stems:
                    for suffix in ['.json', '.safetensors']:
                        (directory / (stem + suffix)).touch()
            dino.mkdir()
            (dino / 'config.json').touch()
            (dino / 'model.safetensors').touch()
            norm = {'mean': list(range(32)), 'std': [1] * 32}
            template = root / 'template.json'
            template.write_text(json.dumps({'name': 'Trellis2ImageTo3DPipeline', 'args': {
                'shape_slat_normalization': copy.deepcopy(norm), 'tex_slat_normalization': copy.deepcopy(norm)}}))
            output = root / 'runtime/pipeline.json'
            # Hash matching is mocked ONLY inside this unit test with empty fixtures.
            lock = json.loads(reproduce.LOCK.read_text())
            with patch.object(reproduce, 'digest', return_value=lock['dinov3_model_sha256']):
                reproduce.configure(template, weights, dino, output)
                args = json.loads(output.read_text())['args']
                self.assertEqual(args['shape_slat_normalization'], norm)
                self.assertEqual(args['tex_slat_normalization'], norm)
                self.assertTrue(args['low_vram'])
                self.assertEqual(len(args['models']), 8)
                with self.assertRaises(FileExistsError):
                    reproduce.configure(template, weights, dino, output)
                (dino / 'config.json').unlink()
                with self.assertRaisesRegex(ValueError, 'Missing required'):
                    reproduce.configure(template, weights, dino, root / 'bad.json')


if __name__ == '__main__':
    unittest.main()
