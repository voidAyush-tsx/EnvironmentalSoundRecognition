"""Model integration tests use fake upstream modules and tiny tensor checkpoints."""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from src.config import CNN14_FRONTEND, EXPECTED_PANN_COMMIT
from src.pann_adapter import load_backbone, new_backbone


class PannAdapterTests(unittest.TestCase):
    def test_private_upstream_import_does_not_shadow_existing_models_or_utils(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "pytorch"
            directory.mkdir()
            (directory / "pytorch_utils.py").write_text("TOKEN = 'pinned-utils'\n", encoding="utf-8")
            (directory / "models.py").write_text(
                "from pytorch_utils import TOKEN\nclass Cnn14:\n"
                "    def __init__(self, **kwargs):\n        self.args = kwargs\n        self.token = TOKEN\n",
                encoding="utf-8")
            models, utils = types.ModuleType("models"), types.ModuleType("pytorch_utils")
            path_before = sys.path[:]
            with patch.dict(sys.modules, {"models": models, "pytorch_utils": utils}), \
                 patch("src.pann_adapter.actual_source_commit", return_value=EXPECTED_PANN_COMMIT):
                backbone = new_backbone(root)
                self.assertEqual(backbone.args, CNN14_FRONTEND)
                self.assertEqual(backbone.token, "pinned-utils")
                self.assertIs(sys.modules["models"], models)
                self.assertIs(sys.modules["pytorch_utils"], utils)
            self.assertEqual(sys.path, path_before)

    def test_wrong_revision_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pytorch").mkdir()
            for name in ("models.py", "pytorch_utils.py"):
                (root / "pytorch" / name).write_text("raise AssertionError('must not import')\n")
            with patch("src.pann_adapter.actual_source_commit", return_value="wrong-revision"):
                with self.assertRaisesRegex(RuntimeError, "commit mismatch"):
                    new_backbone(root)

    def test_restricted_strict_backbone_checkpoint_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "backbone.pth"
            original = nn.Linear(2, 3)
            torch.save({"model": original.state_dict()}, path)
            with patch("src.pann_adapter.new_backbone", return_value=nn.Linear(2, 3)), \
                 patch("src.pann_adapter.torch.load", wraps=torch.load) as loader:
                restored = load_backbone(weights=path, source=tmp)
                loader.assert_called_once_with(path.resolve(), map_location="cpu", weights_only=True)
            for name, value in original.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[name]))
            torch.save({"model": {"wrong": torch.zeros(1)}}, path)
            with patch("src.pann_adapter.new_backbone", return_value=nn.Linear(2, 3)):
                with self.assertRaises(RuntimeError):
                    load_backbone(weights=path, source=tmp)


if __name__ == "__main__":
    unittest.main()
