"""Exact packed values, activation gradients, PEFT master precision and entropy."""
import unittest

import torch
from torch import nn

from verl.extensions.gemma.packed_base import FrozenPackedLinear, unpack_weight, validate_scheme
from verl.extensions.gemma.selected_head import entropy_totals


def fixture():
    q = (torch.arange(3 * 64).reshape(3, 64) % 16 - 8).to(torch.int32)
    shifts = torch.arange(0, 32, 4, dtype=torch.int64)
    packed = (((q.long() + 8).reshape(3, 8, 8) << shifts).sum(-1)).to(torch.int32)
    scale = torch.tensor([[0.125, 0.25], [0.5, 1.0], [2.0, 0.0625]], dtype=torch.bfloat16)
    return q, packed, scale


class PackedBaseTests(unittest.TestCase):
    def test_all_signed_values_and_groups_exact(self):
        q, packed, scale = fixture()
        got = unpack_weight(packed, scale.view(torch.uint8), 64, torch.float32)
        expected = q.float() * scale.float().repeat_interleave(32, dim=1)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        layer = FrozenPackedLinear(packed, scale, [3, 64]).float()
        self.assertEqual(layer.weight_scale_bits.dtype, torch.uint8)
        self.assertTrue(torch.equal(layer.weight_scale_bits.view(torch.bfloat16), scale))
        self.assertEqual(sum(x.numel() for x in layer.parameters()), 0)

    def test_activation_gradient_and_no_saved_dense_weight(self):
        q, packed, scale = fixture()
        layer = FrozenPackedLinear(packed, scale, [3, 64])
        x = torch.randn(2, 5, 64, requires_grad=True)
        reference = x.detach().clone().requires_grad_()
        w = q.float() * scale.float().repeat_interleave(32, dim=1)
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t: (saved.append(t), t)[1], lambda t: t):
            actual = layer(x)
        expected = torch.nn.functional.linear(reference, w)
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
        self.assertEqual([t.dtype for t in saved], [torch.int32, torch.uint8])
        self.assertFalse(any(t.shape == w.shape for t in saved))

    def test_peft_update_changes_only_fp32_adapters(self):
        from peft import LoraConfig, get_peft_model

        _, packed, scale = fixture()
        # The real Gemma includes ordinary unquantized embeddings and norms.
        model = nn.Sequential(nn.LayerNorm(64), FrozenPackedLinear(packed.clone(), scale.clone(), [3, 64]))
        adapted = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["1"]))
        params = [p for p in adapted.parameters() if p.requires_grad]
        self.assertEqual({p.dtype for p in params}, {torch.float32})
        before = {k: v.clone() for k, v in adapted.named_buffers()}
        optimizer = torch.optim.AdamW(params, lr=1e-5)
        x = torch.randn(4, 64)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = adapted(x).float().square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is not None and p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all() for p in params))
        self.assertTrue(any(p.grad.abs().sum() > 0 for p in params))
        optimizer.step()
        for name, value in adapted.named_buffers():
            self.assertTrue(torch.equal(value, before[name]))

    def test_entropy_excludes_padding_and_is_chunk_invariant(self):
        logits = torch.tensor([[0., 0., 0.], [100., -100., -100.], [1., 2., 3.]])
        mask = torch.tensor([True, False, True])
        distribution = torch.distributions.Categorical(logits=logits[mask])
        for chunk in [1, 2, 32]:
            totals = entropy_totals(logits, mask, chunk)
            self.assertEqual(totals[1].item(), 2)
            torch.testing.assert_close(totals[0], distribution.entropy().double().sum())
        self.assertEqual(entropy_totals(logits, ~torch.ones(3, dtype=torch.bool)).sum().item(), 0)

    def test_unknown_quantization_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_scheme({"quantization_config": {"quant_method": "bitsandbytes"}})
        _, packed, scale = fixture()
        with self.assertRaises(ValueError):
            FrozenPackedLinear(packed, scale.float(), [3, 64])


if __name__ == "__main__":
    unittest.main()
