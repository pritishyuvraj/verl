"""Frozen official group-32 INT4 weights with differentiable BF16 activations.

No requantization: packed integers and scale bits come directly from safetensors.
Only the current linear's weight is unpacked for a matmul. Backward reconstructs
that weight for dInput; autograd never retains a dense base weight or computes dW.
This is a memory-saving reference implementation, not a fused INT4 training kernel.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def unpack_weight(packed, scale_bits, in_features, dtype):
    # compressed-tensors packs unsigned nibbles: signed q = nibble - 8.
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    q = ((packed.unsqueeze(-1) >> shifts) & 15).flatten(1)[:, :in_features]
    q = (q - 8).to(dtype)
    scales = scale_bits.view(torch.bfloat16).to(dtype)
    return (q.reshape(q.shape[0], -1, 32) * scales.unsqueeze(-1)).reshape(q.shape)


class _FrozenLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, scale_bits, in_features):
        ctx.save_for_backward(packed, scale_bits)
        ctx.in_features = in_features
        weight = unpack_weight(packed, scale_bits, in_features, x.dtype)
        return F.linear(x, weight)

    @staticmethod
    def backward(ctx, grad_output):
        packed, scale_bits = ctx.saved_tensors
        weight = unpack_weight(packed, scale_bits, ctx.in_features, grad_output.dtype)
        return grad_output.matmul(weight), None, None, None


class FrozenPackedLinear(nn.Linear):
    """nn.Linear-compatible PEFT base; merging adapters is intentionally unsupported."""

    def __init__(self, packed, scale, shape):
        nn.Module.__init__(self)
        self.out_features, self.in_features = map(int, shape)
        if self.in_features % 32 or packed.dtype != torch.int32 or scale.dtype != torch.bfloat16:
            raise ValueError("Only official BF16-scale, symmetric group32 INT4 is qualified")
        if tuple(packed.shape) != (self.out_features, self.in_features // 8):
            raise ValueError("Packed weight shape differs from original linear")
        if tuple(scale.shape) != (self.out_features, self.in_features // 32):
            raise ValueError("Scale shape differs from group32 weights")
        if not torch.isfinite(scale).all() or (scale < 0).any():
            raise ValueError("Nonfinite or negative quantization scale")
        self.register_buffer("weight_packed", packed.detach().contiguous())
        # Preserve original scale bits even when model.to(float32) is called.
        self.register_buffer("weight_scale_bits", scale.detach().contiguous().view(torch.uint8))
        self.register_parameter("bias", None)

    @property
    def weight(self):
        # PEFT reads weight.device/dtype to initialize adapters. This is metadata,
        # never a matrix or Parameter; an attempted merge fails by shape mismatch.
        return torch.empty(0, dtype=torch.float32, device=self.weight_packed.device)

    def forward(self, x):
        dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
        return _FrozenLinear.apply(x.to(dtype), self.weight_packed, self.weight_scale_bits, self.in_features)


def validate_scheme(config):
    q = config.get("quantization_config", {})
    groups = q.get("config_groups", {})
    if q.get("quant_method") != "compressed-tensors" or q.get("format") != "pack-quantized" or len(groups) != 1:
        raise ValueError("Expected the official packed compressed-tensors checkpoint")
    group = next(iter(groups.values()))
    w = group.get("weights", {})
    expected = {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "group", "group_size": 32}
    if any(w.get(k) != v for k, v in expected.items()) or w.get("actorder") is not None:
        raise ValueError("Unsupported quantization values/scales; no automatic conversion")
    if group.get("input_activations") is not None or group.get("output_activations") is not None:
        raise ValueError("Only weight-only quantization is supported")
    return q


def load_model(*, auto_class, model_config, dtype):
    """Load exact official values before PEFT/FSDP; no dense text-linear allocation."""
    from accelerate import init_empty_weights
    from safetensors import safe_open

    root = Path(model_config.local_path)
    raw_config = json.loads((root / "config.json").read_text())
    scheme = validate_scheme(raw_config)
    if dtype != torch.float32 or model_config.lora_rank <= 0:
        raise ValueError("Frozen packed training requires FP32 masters and trainable LoRA")
    config = copy.deepcopy(model_config.hf_config)
    if hasattr(config, "quantization_config"):
        del config.quantization_config
    with init_empty_weights(include_buffers=False):
        model = auto_class.from_config(config, torch_dtype=dtype, trust_remote_code=False)
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise ValueError("No checkpoint tensors")
    seen, packed_names, normal = set(), set(), {}
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as tensors:
            keys = set(tensors.keys())
            if keys & seen:
                raise ValueError("Duplicate tensor names across checkpoint shards")
            seen |= keys
            for key in sorted(keys):
                if not key.endswith(".weight_packed"):
                    continue
                name = key.removesuffix(".weight_packed")
                if not name.startswith("model.language_model."):
                    raise ValueError("Unexpected non-text packed module")
                if name + ".weight_zero_point" in keys or name + ".weight_g_idx" in keys:
                    raise ValueError("Unexpected zero points or reordered groups")
                old = model.get_submodule(name)
                shape = tensors.get_tensor(name + ".weight_shape").tolist()
                if not isinstance(old, nn.Linear) or old.bias is not None or shape != list(old.weight.shape):
                    raise ValueError(f"Checkpoint/module mismatch: {name}")
                replacement = FrozenPackedLinear(tensors.get_tensor(key), tensors.get_tensor(name + ".weight_scale"), shape)
                parent, leaf = name.rsplit(".", 1)
                setattr(model.get_submodule(parent), leaf, replacement)
                packed_names.add(name)
            expected = model.state_dict()
            for key in keys:
                if key in expected and not key.endswith(".weight_packed"):
                    value = tensors.get_tensor(key)
                    if value.shape != expected[key].shape:
                        raise ValueError(f"Unquantized tensor shape mismatch: {key}")
                    normal[key] = value.to(dtype) if value.is_floating_point() else value
    if not packed_names:
        raise ValueError("No frozen INT4 layers were loaded")
    result = model.load_state_dict(normal, strict=False, assign=True)
    allowed_missing = {name + suffix for name in packed_names for suffix in (".weight_packed", ".weight_scale_bits")}
    if set(result.missing_keys) != allowed_missing or result.unexpected_keys:
        raise ValueError(f"Incomplete official weights: {result}")
    model.tie_weights()
    if any(p.is_meta for p in model.parameters()) or any(b.is_meta for b in model.buffers()):
        raise ValueError("Meta tensors remain after exact checkpoint loading")
    model.requires_grad_(False)
    model._gemma_packed_base = {
        "source": str(root), "layers": len(packed_names), "quantization": scheme,
        "packed_bytes": sum(m.weight_packed.numel() * 4 + m.weight_scale_bits.numel() * m.weight_scale_bits.element_size() for m in model.modules() if isinstance(m, FrozenPackedLinear)),
        "requantized": False, "dense_frozen_text_linears_retained": False,
    }
    return model
