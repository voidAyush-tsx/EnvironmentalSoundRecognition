"""Fast CPU smoke tests for Stage 4 train/validation aggregation."""
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.finetune_train import train_epoch, validate_epoch


class LoopTests(unittest.TestCase):
    def test_train_and_validate_with_accumulation(self):
        torch.manual_seed(7)
        model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 10))
        waves = torch.randn(5, 4)
        labels = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        loader = DataLoader(TensorDataset(waves, labels), batch_size=1)
        optim = torch.optim.SGD(model.parameters(), lr=0.05)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        before = model[0].weight.detach().clone()
        tr = train_epoch(model, loader, optim, scaler, torch.device("cpu"), False, 2, 3)
        self.assertEqual(tr["samples"], 3)
        self.assertEqual(tr["batches"], 3)
        self.assertGreater((before-model[0].weight).abs().max().item(), 0)
        val = validate_epoch(model, loader, torch.device("cpu"), False, 2)
        self.assertEqual(val["samples"], 2)
        self.assertIn("macro_f1", val)
        self.assertGreaterEqual(val["macro_f1"], 0)
        self.assertLessEqual(val["macro_f1"], 1)


if __name__ == "__main__":
    unittest.main()
