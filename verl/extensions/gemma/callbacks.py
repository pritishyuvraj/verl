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


class ExportAdapter:
    """Export after the native resumable checkpoint, collectively on all ranks.

    Native model/optimizer/RNG/dataloader shards remain authoritative for resume.
    The separate adapter directory is for inference, never optimizer recovery.
    """

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
