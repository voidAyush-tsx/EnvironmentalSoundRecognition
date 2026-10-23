"""CPU regression tests: zero fc1 weight gradient is not a graph failure."""
import unittest
import pandas as pd
import torch
from torch import nn
from src.gradient_audit import candidate_indices, gradient_measure, feature_statistics
from src.finetune_train import train_epoch
from torch.utils.data import DataLoader, TensorDataset

class GradientAuditTests(unittest.TestCase):
    def test_deterministic_training_only_indices(self):
        frame = pd.DataFrame({"classID": [3,3,7,1,0,7,2,8,5,4,6,9], "fold": [3]*12})
        indices = candidate_indices(frame, 6)
        self.assertEqual(indices, [0,4,3,6,9,8])
        self.assertTrue(all(frame.iloc[i].fold == 3 for i in indices))

    def test_positive_head_grad_but_zero_fc1_weight_grad_when_input_zero(self):
        torch.manual_seed(4)
        fc1 = nn.Linear(4,4)
        head = nn.Linear(4,10)
        with torch.no_grad():
            fc1.bias.fill_(1.0)  # active ReLU despite zero input features
        inputs = torch.zeros(1,4)
        y = head(torch.relu(fc1(inputs)))
        loss = nn.functional.cross_entropy(y, torch.tensor([3]))
        loss.backward()
        self.assertEqual(gradient_measure(fc1.weight), ("zero", 0.0))
        self.assertEqual(gradient_measure(fc1.bias)[0], "nonzero")
        self.assertEqual(gradient_measure(head.weight)[0], "nonzero")

    def test_normal_nonzero_fc1_gradient(self):
        fc = nn.Linear(4,2)
        x = torch.randn(1,4)
        y = fc(x)
        y.sum().backward()
        self.assertEqual(gradient_measure(fc.weight)[0], "nonzero")
        self.assertGreater(feature_statistics(x)["max_abs"],0)

    def test_train_epoch_rejects_inactive_pretrained_fc1(self):
        class FakeBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4,4)
            def forward(self, x):
                return torch.relu(self.fc1(x))
        class ToyFineTuner(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = FakeBackbone()
                self.classifier = nn.Linear(4,10)
            def forward(self,x):
                return self.classifier(self.backbone(x))
        model = ToyFineTuner()
        with torch.no_grad():
            model.backbone.fc1.bias.fill_(1.0)
        loader = DataLoader(TensorDataset(torch.zeros(2,4),torch.tensor([1,3])),batch_size=1)
        opt = torch.optim.AdamW(model.parameters(),lr=0.001)
        scaler = torch.amp.GradScaler("cuda",enabled=False)
        with self.assertRaisesRegex(RuntimeError, "No nonzero training gradients.*fc1"):
            train_epoch(model,loader,opt,scaler,torch.device("cpu"),"off",1,0)

    def test_reject_empty_candidate_count(self):
        with self.assertRaises(ValueError):
            candidate_indices(pd.DataFrame({"classID":[3]}),0)

if __name__ == "__main__":
    unittest.main()
