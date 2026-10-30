import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from verl.extensions.gemma.rl_sync import enabled, weights


class AdapterSyncTest(unittest.TestCase):
    def test_frozen_worker_skips_only_base_refit_hooks(self):
        from verl.extensions.gemma.rl_worker import PackedGemmaLoRAWorkerExtension
        from verl.workers.rollout.vllm_rollout import utils
        config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="gemma4")),
                                 lora_config=object())
        with patch.object(utils, "set_death_signal"), patch.object(utils.VLLMHijack, "hijack") as lora, \
             patch.object(utils, "is_quantized_model", return_value=True), \
             patch.object(utils, "configure_fp8_ignored_layers_match_mode"), \
             patch.object(utils, "apply_vllm_quant_patches") as refit:
            PackedGemmaLoRAWorkerExtension(vllm_config=config)
            lora.assert_called_once()
            refit.assert_not_called()
            utils.vLLMColocateWorkerExtension(vllm_config=config)
            refit.assert_called_once()

    def test_frozen_worker_rejects_base_update(self):
        from verl.extensions.gemma.rl_worker import PackedGemmaLoRAWorkerExtension
        worker = object.__new__(PackedGemmaLoRAWorkerExtension)
        with self.assertRaisesRegex(ValueError, "must never"):
            worker.update_weights_from_ipc(peft_config={"r":32}, base_sync_done=False)
        with self.assertRaisesRegex(ValueError, "only adapter"):
            worker._update_weights([("model.weight", torch.ones(1))], {"r":32}, True)

    def test_frozen_base_is_never_transferred(self):
        engine = Mock()
        engine.get_per_tensor_param.return_value = (
            [("model.layer.weight", torch.ones(2))], {"r": 32})
        with self.assertRaisesRegex(RuntimeError, "Frozen base"):
            list(weights(engine))

    def test_only_adapter_values_preserved(self):
        value = torch.randn(3, 2)
        engine = Mock()
        engine.get_per_tensor_param.return_value = (
            [("model.layer.lora_A.weight", value)], {"r": 32})
        result = list(weights(engine))
        self.assertIs(result[0][1], value)
        engine.get_per_tensor_param.assert_called_once_with(base_sync_done=True)

    def test_unrelated_models_and_merging(self):
        self.assertFalse(enabled(SimpleNamespace()))
        config = SimpleNamespace(rl_head_chunk_size=32,
            weight_loader="verl.extensions.gemma.rl_head.load_model",lora_rank=32,lora={"merge":True})
        with self.assertRaises(ValueError):
            enabled(config)
