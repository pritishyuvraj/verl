"""Frozen Gemma INT4 serving: native loading, adapter-only synchronization."""
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension


class PackedGemmaLoRAWorkerExtension(vLLMColocateWorkerExtension):
    # The checkpoint's INT4 kernels load normally. No FP4/FP8 base refit occurs.
    frozen_base_adapter_only = True

    def __new__(cls, **kwargs):
        config = kwargs["vllm_config"]
        if config.model_config.hf_config.model_type != "gemma4":
            raise ValueError("Frozen adapter-only worker requires Gemma 4")
        if config.lora_config is None:
            raise ValueError("Frozen adapter-only worker requires LoRA")
        return super().__new__(cls, **kwargs)

    def update_weights_from_ipc(self, peft_config=None, base_sync_done=False, use_shm=False):
        if not peft_config or not base_sync_done:
            raise ValueError("Frozen Gemma base must never enter weight refit")
        return super().update_weights_from_ipc(peft_config, base_sync_done, use_shm)

    def _update_weights(self, weights, peft_config, base_sync_done):
        if not peft_config or not base_sync_done or any(
            ".lora_A." not in name and ".lora_B." not in name for name, _ in weights
        ):
            raise ValueError("Frozen Gemma synchronization accepts only adapter tensors")
        return super()._update_weights(weights, peft_config, base_sync_done)
