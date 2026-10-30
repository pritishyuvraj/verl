import unittest

import torch
from tensordict import TensorDict

from verl.extensions.gemma.rl_head import _forward, chunk_statistics, prepare_inputs, restore_outputs


class PolicyHeadTest(unittest.TestCase):
    def test_logprob_entropy_and_gradient_match_dense(self):
        torch.manual_seed(42)
        head = torch.nn.Linear(7, 31, bias=False).double().requires_grad_(False)
        original = torch.randn(13, 7, dtype=torch.float64)
        labels = torch.randint(31, (13,))
        for cap in (None, 30.0):
            a = original.clone().requires_grad_(True)
            b = original.clone().requires_grad_(True)
            logits = head(a)
            if cap:
                logits = torch.tanh(logits / cap) * cap
            logp = torch.log_softmax(logits.float() / 0.8, -1)
            expected = logp.gather(-1, labels[:, None]).squeeze(-1)
            entropy = -(logp.exp() * logp).sum(-1)
            actual, actual_entropy = chunk_statistics(head, b, labels, chunk_size=3, temperature=0.8, softcap=cap)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(actual_entropy, entropy)
            (expected.sum() + 0.03 * entropy.sum()).backward()
            (actual.sum() + 0.03 * actual_entropy.sum()).backward()
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5)

    def test_response_predecessors_and_environment_mask(self):
        def nested(v):
            return torch.nested.nested_tensor([torch.tensor(v)], layout=torch.jagged)
        data = TensorDict({"input_ids": nested([8, 9, 10, 11, 12]),
                           "responses": nested([10, 11, 12]),
                           "response_mask": nested([1, 0, 1])}, batch_size=[1])
        inputs = {}
        positions = prepare_inputs(data, inputs, {"temperature": torch.tensor([1.0])})
        self.assertEqual(positions.tolist(), [1, 3])
        self.assertEqual(inputs["gemma_rl_labels"].tolist(), [10, 12])
        from types import SimpleNamespace
        out = restore_outputs(SimpleNamespace(gemma_log_probs=torch.tensor([-1., -2.]),
                                              gemma_entropy=torch.tensor([3., 4.])), data, positions)
        self.assertEqual(out["log_probs"].values().tolist(), [0., -1., 0., -2., 0.])

    def test_empty_response_refused(self):
        with self.assertRaises(ValueError):
            chunk_statistics(torch.nn.Linear(7, 31), torch.empty(0, 7), torch.empty(0, dtype=torch.long))

    def test_native_padding_forward_backward_contributes_zero(self):
        from types import SimpleNamespace
        from verl.trainer.ppo.padding_utils import construct_minimal_padding_template
        sample, tag = construct_minimal_padding_template({}, {}, eos_token_id=1)
        self.assertTrue(tag['is_padding'])
        data = TensorDict({key: torch.nested.nested_tensor([sample[key]], layout=torch.jagged)
                           for key in ('input_ids', 'responses', 'response_mask')}, batch_size=[1])
        inputs = {}
        positions = prepare_inputs(data, inputs, {'temperature': torch.tensor([1.0])})
        self.assertEqual(positions.numel(), 0)
        decoder = torch.nn.Embedding(8, 7)
        model = SimpleNamespace(model=lambda **kwargs: SimpleNamespace(
            last_hidden_state=decoder(sample['input_ids'].unsqueeze(0))))
        raw = _forward(model, **inputs)
        result = restore_outputs(raw, data, positions)
        loss = result['log_probs'].values().sum() + result['entropy'].values().sum()
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertIsNotNone(decoder.weight.grad)
        self.assertEqual(decoder.weight.grad.abs().sum().item(), 0)

    def test_bad_masks_refused(self):
        for mask in ([1, 2], [1]):
            data = TensorDict({key: torch.nested.nested_tensor([torch.tensor(value)], layout=torch.jagged)
                for key, value in {'input_ids':[1,2,3], 'responses':[2,3], 'response_mask':mask}.items()}, batch_size=[1])
            with self.assertRaisesRegex(ValueError, 'Response mask'):
                prepare_inputs(data, {}, {'temperature':torch.tensor([1.0])})


if __name__ == "__main__":
    unittest.main()
