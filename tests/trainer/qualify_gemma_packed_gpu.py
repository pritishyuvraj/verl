"""One bounded actual-model forward/backward; no optimizer update or candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForImageTextToText

from verl.extensions.gemma.packed_base import FrozenPackedLinear, load_model


def digest_buffers(model):
    h = hashlib.sha256()
    for name, module in model.named_modules():
        if isinstance(module, FrozenPackedLinear):
            h.update(name.encode())
            for tensor in (module.weight_packed, module.weight_scale_bits):
                h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    started = time.monotonic()
    torch.manual_seed(20261028)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True, attn_implementation="sdpa")
    model = load_model(auto_class=AutoModelForImageTextToText,
                       model_config=SimpleNamespace(local_path=args.model, hf_config=config, lora_rank=32),
                       dtype=torch.float32)
    metadata = model._gemma_packed_base
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=32, lora_alpha=64, bias="none",
                          target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"))
    before = digest_buffers(model)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train().to("cuda:0")
    ids = torch.arange(100, 132, device="cuda:0").unsqueeze(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        loss = torch.nn.functional.cross_entropy(output.logits[:, :-1].float().flatten(0, 1), ids[:, 1:].flatten())
    loss.backward()
    torch.cuda.synchronize()
    params = [p for p in model.parameters() if p.requires_grad]
    assert params and all(p.dtype == torch.float32 for p in params)
    assert all(p.grad is not None and p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all() for p in params)
    nonzero = sum(bool(p.grad.abs().sum() > 0) for p in params)
    assert nonzero > 0 and torch.isfinite(loss)
    after = digest_buffers(model)
    assert before == after
    receipt = {"status": "pass", "seconds": time.monotonic() - started, "loss": loss.item(),
               "packed_layers": metadata["layers"], "packed_bytes": metadata["packed_bytes"],
               "packed_buffer_sha256_before": before, "packed_buffer_sha256_after": after,
               "trainable_elements": sum(p.numel() for p in params), "trainable_dtype": "FP32",
               "gradient_dtype": "FP32", "nonzero_gradient_tensors": nonzero,
               "gradient_tensors": len(params), "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
               "optimizer_updates": 0, "candidate_created": False}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
