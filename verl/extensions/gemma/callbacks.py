"""Configurable Gemma setup and portable PEFT checkpoint export."""

import os
import tempfile
from pathlib import Path


class SelectedHead:
    def __init__(self, fp32_lm_head=True):
        self.fp32_lm_head = fp32_lm_head

    def on_train_start(self, trainer):
        import torch

        from .selected_head import install_selected_head

        torch.backends.cuda.matmul.allow_tf32 = False
        install_selected_head(trainer.engine, fp32_lm_head=self.fp32_lm_head)
        trainer.engine._gemma_validation_entropy = trainer.config.trainer.get("validation_entropy", False)


class ExportAdapter:
    """Export after the native resumable checkpoint, collectively on all ranks.

    Native model/optimizer/RNG/dataloader shards remain authoritative for resume.
    The separate adapter directory is for inference, never optimizer recovery.
    """

    def __init__(self, dtype="bfloat16"):
        if dtype not in ("bfloat16", "float32"):
            raise ValueError("Adapter export dtype must be bfloat16 or float32")
        self.dtype = dtype

    def on_checkpoint(self, trainer, step, checkpoint_path):
        import torch
        import torch.distributed as dist
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file

        module = trainer.engine.module
        state = {}
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                value = parameter.detach()
                value = value.full_tensor() if hasattr(value, "full_tensor") else value
                if trainer.rank == 0:
                    state[name] = value.cpu().contiguous()

        # Propagate rank-zero IO/validation errors instead of stranding other ranks
        # at a success barrier. torchrun still owns failures during the gathers.
        error = [None]
        if trainer.rank == 0:
            try:
                destination = Path(checkpoint_path) / "adapter"
                if destination.exists():
                    raise FileExistsError(destination)
                adapter = get_peft_model_state_dict(module, state_dict=state)
                if not adapter or not all(torch.isfinite(value).all() for value in adapter.values()):
                    raise ValueError("Empty or nonfinite adapter export")
                # Cast only the inference copy. Native optimizer/master state is
                # preserved at training precision for the next resume phase.
                adapter = {
                    name: value.to(dtype=getattr(torch, self.dtype)).contiguous() for name, value in adapter.items()
                }
                if not all(torch.isfinite(value).all() for value in adapter.values()):
                    raise ValueError("Nonfinite adapter after export dtype conversion")
                with tempfile.TemporaryDirectory(prefix=".adapter-", dir=checkpoint_path) as temporary:
                    path = Path(temporary)
                    save_file(adapter, path / "adapter_model.safetensors", metadata={"format": "pt"})
                    module.peft_config["default"].save_pretrained(path)
                    (path / "training_step.txt").write_text(str(step) + "\n")
                    os.rename(path, destination)
            except Exception as exception:
                error[0] = f"{type(exception).__name__}: {exception}"
        dist.broadcast_object_list(error, src=0)
        if error[0] is not None:
            raise RuntimeError(f"Portable adapter export failed: {error[0]}")


class PackedPrecision:
    """Verify the live optimizer contract, including after native resume."""

    def on_train_start(self, trainer):
        import json
        import torch

        from .packed_base import FrozenPackedLinear

        module = trainer.engine.module
        packed = [m for m in module.modules() if isinstance(m, FrozenPackedLinear)]
        trainable = [p for p in module.parameters() if p.requires_grad]
        if not packed or not trainable or any(p.dtype != torch.float32 for p in trainable):
            raise ValueError("Packed base and FP32 LoRA master contract not met")
        if any(m.weight_packed.dtype != torch.int32 or m.weight_scale_bits.dtype != torch.uint8 for m in packed):
            raise ValueError("Packed weight or original scale storage changed")
        original = trainer.engine.optimizer_step
        checked = [False]

        def step():
            gradients = [p.grad for p in trainable if p.grad is not None]
            if not gradients or any(g.dtype != torch.float32 for g in gradients):
                raise ValueError("LoRA gradients are not FP32 before optimizer update")
            if not checked[0]:
                print(json.dumps({"precision_verified_rank": trainer.rank, "master_dtype": "FP32",
                                  "gradient_dtype": "FP32", "packed_layers": len(packed)}), flush=True)
                checked[0] = True
            return original()

        trainer.engine.optimizer_step = step
