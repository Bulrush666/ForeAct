import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moge3_depth.config import load_config, target_frame_index, target_frame_indices


class ForeActLayoutTests(unittest.TestCase):
    def test_all_recipes_load(self):
        for path in (ROOT / 'configs').glob('*.yaml'):
            with self.subTest(path=path.name):
                config = load_config(path)
                self.assertTrue(Path(config['_config_path']).is_absolute())

    def test_cli_defaults_are_absolute_and_dataset_specific(self):
        for path in sorted(ROOT.glob('*.py')):
            if path.name == 'check_libero_setup.py':
                continue
            with self.subTest(path=path.name):
                spec = importlib.util.spec_from_file_location('_moge_test_' + path.stem, path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                with patch.object(sys, 'argv', [path.name]):
                    args = module.parse_args()
                self.assertTrue(args.config.is_absolute())
                self.assertTrue(args.config.is_file())
                if 'fractal' in path.name:
                    self.assertEqual(args.config.name, 'fractal_next1.yaml')
                if 'libero' in path.name:
                    self.assertEqual(args.config.name, 'libero_next1.yaml')

    def test_next_frame_and_chunk_end_padding(self):
        self.assertEqual(target_frame_index(0, 8, 1), 1)
        self.assertEqual(target_frame_index(7, 8, 1), 7)
        self.assertEqual(target_frame_indices(1, 1), [0])
        self.assertEqual(target_frame_indices(8, 1), list(range(1, 8)))
        self.assertEqual(target_frame_indices(8, 16), [7])
        self.assertEqual(target_frame_indices(26, 16), list(range(16, 26)))


if __name__ == '__main__':
    unittest.main()
