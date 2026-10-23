"""Lightweight model-freeze tests; never downloads a 1.4 GB checkpoint."""
import unittest

import torch
from torch import nn
from src.finetune_model import (
    CNN14ForUrbanSound, parameter_summary, optimizer_for_model,
    trainable_state, load_trainable_state,
)


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.bn2 = nn.BatchNorm2d(2)


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(2)
        for i in range(1, 7):
            setattr(self, f"conv_block{i}", ToyBlock())
        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, 527)

    def forward(self, x):
        # Uses fc1 so the test can exercise backprop through a small batch.
        h = torch.ones((x.shape[0], 2048), device=x.device)
        h = self.fc1(h)
        h = self.conv_block6.bn1(h.reshape(-1, 2, 1, 1024)).reshape(-1, 2048)
        return {"embedding": h, "clipwise_output": self.fc_audioset(h).sigmoid()}


class FreezePolicyTests(unittest.TestCase):
    def setUp(self):
        self.model = CNN14ForUrbanSound(ToyBackbone(), config="head")

    def test_head_only(self):
        self.assertTrue(all(not p.requires_grad for p in self.model.backbone.parameters()))
        self.assertTrue(all(p.requires_grad for p in self.model.classifier.parameters()))
        self.model.train()
        self.assertFalse(self.model.backbone.bn0.training)
        self.assertFalse(self.model.backbone.conv_block6.bn1.training)

    def test_fc1_only(self):
        self.model.configure("fc1")
        self.assertTrue(self.model.backbone.fc1.weight.requires_grad)
        self.assertFalse(self.model.backbone.conv_block6.conv1.weight.requires_grad)
        self.assertFalse(self.model.backbone.fc_audioset.weight.requires_grad)

    def test_later_conv_trainable_bn_stats_frozen(self):
        self.model.configure("block6")
        self.model.train()
        self.assertTrue(self.model.backbone.conv_block6.conv1.weight.requires_grad)
        self.assertTrue(self.model.backbone.conv_block6.bn1.weight.requires_grad)
        self.assertFalse(self.model.backbone.conv_block5.conv1.weight.requires_grad)
        self.assertFalse(self.model.backbone.conv_block6.bn1.training)
        self.assertTrue(self.model.backbone.spec_augmenter.training if hasattr(self.model.backbone,"spec_augmenter") else True)

    def test_optimizer_exact_parameter_selection(self):
        self.model.configure("block6")
        optim = optimizer_for_model(self.model)
        actual = {id(p) for group in optim.param_groups for p in group["params"]}
        expected = {id(p) for p in self.model.parameters() if p.requires_grad}
        self.assertEqual(actual, expected)
        summary = parameter_summary(self.model)
        self.assertEqual(summary["total"], summary["trainable"] + summary["frozen"])

    def test_trainable_checkpoint_roundtrip(self):
        self.model.configure("fc1")
        saved = trainable_state(self.model)
        with torch.no_grad():
            self.model.classifier.weight.add_(3)
        load_trainable_state(self.model, saved)
        self.assertTrue(torch.equal(self.model.classifier.weight, saved["classifier.weight"]))


if __name__ == "__main__":
    unittest.main()
