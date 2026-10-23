"""No large CNN14 checkpoint needed; tests safe precision policy."""
from unittest import TestCase
from unittest.mock import patch
import torch
from src.precision_policy import choose_precision, autocast_context, assert_finite_logits

class PrecisionTests(TestCase):
    def test_cpu_fallback(self):
        self.assertEqual(choose_precision(torch.device("cpu"), "auto"), "off")
        self.assertEqual(choose_precision(torch.device("cpu"), "off"), "off")
        with self.assertRaises(ValueError):
            choose_precision(torch.device("cpu"), "on")

    def test_cuda_bfloat16_selection(self):
        with patch("src.precision_policy.torch.cuda.is_bf16_supported", return_value=True):
            self.assertEqual(choose_precision(torch.device("cuda"), "auto"), "bf16")
            self.assertEqual(choose_precision(torch.device("cuda"), "on"), "bf16")
            self.assertEqual(choose_precision(torch.device("cuda"), "off"), "off")

    def test_unsupported_cuda_never_enables_fp16(self):
        with patch("src.precision_policy.torch.cuda.is_bf16_supported", return_value=False):
            self.assertEqual(choose_precision(torch.device("cuda"), "auto"), "off")
            with self.assertRaises(RuntimeError):
                choose_precision(torch.device("cuda"), "bf16")

    def test_cpu_noop_autocast_and_nonfinite(self):
        with autocast_context(torch.device("cpu"), "off"):
            assert_finite_logits(torch.zeros(1, 10), phase="unit-test")
            with self.assertRaises(FloatingPointError):
                assert_finite_logits(torch.full((1, 10), float("nan")), phase="unit-test")
