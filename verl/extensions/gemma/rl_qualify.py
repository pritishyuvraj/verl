"""One bounded packed-Gemma policy backward check; never takes an optimizer step."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer
    from verl.extensions.gemma.rl_head import load_model
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    start = time.monotonic()
    config = AutoConfig.from_pretrained(a.model, attn_implementation="sdpa")
    settings = SimpleNamespace(local_path=a.model, hf_config=config, lora_rank=32, rl_head_chunk_size=16)
    base = load_model(auto_class=AutoModelForImageTextToText, model_config=settings, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, a.adapter, is_trainable=True).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ids = tok("Find the defect, edit the implementation, run tests, and submit the patch.", return_tensors="pt").input_ids.cuda()
    selected = torch.arange(1, ids.shape[1]-1, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                    gemma_rl_positions=selected, gemma_rl_labels=ids[0, selected+1])
        loss = -out.gemma_log_probs.mean()
    loss.backward()
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in n or p.dtype != torch.float32 for n,p in trainable):
        raise RuntimeError("Expected only FP32 LoRA masters")
    gradients = [p.grad for _,p in trainable if p.grad is not None]
    if not gradients or any(g.dtype != torch.float32 or not torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("Missing, non-FP32, or nonfinite LoRA gradients")
    result = {"status": "pass", "optimizer_updates": 0, "loss": loss.item(),
              "entropy_nats": out.gemma_entropy.mean().item(), "gradient_tensors": len(gradients),
              "nonzero_gradient_tensors": sum(bool(g.count_nonzero()) for g in gradients),
              "trainable_parameters": sum(p.numel() for _,p in trainable),
              "adapter_sha256": hashlib.sha256((Path(a.adapter)/"adapter_model.safetensors").read_bytes()).hexdigest(),
              "seconds": time.monotonic()-start, "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30}
    with open(a.output,"x") as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
