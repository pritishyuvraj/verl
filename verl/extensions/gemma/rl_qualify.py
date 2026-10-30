"""Packed-Gemma backward checks; optional discarded two-rank optimizer qualification."""
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
    p.add_argument("--distributed-update", action="store_true")
    p.add_argument("--saved-response", help="Complete saved response JSON; no generation or truncation")
    a = p.parse_args()
    start = time.monotonic()
    if a.distributed_update:
        import os
        import torch.distributed as dist
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group('nccl')
        if dist.get_world_size() != 2 or not a.saved_response:
            raise ValueError('Distributed qualification requires two ranks and a saved response')
    config = AutoConfig.from_pretrained(a.model, attn_implementation="sdpa")
    settings = SimpleNamespace(local_path=a.model, hf_config=config, lora_rank=32, rl_head_chunk_size=16)
    base = load_model(auto_class=AutoModelForImageTextToText, model_config=settings, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, a.adapter, is_trainable=True).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if a.distributed_update:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy
        from verl.utils.fsdp_utils import apply_fsdp2, fsdp2_clip_grad_norm_
        apply_fsdp2(model, {'mesh':init_device_mesh('cuda',(2,)), 'reshard_after_forward':True,
            'mp_policy':MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)}, {})
        record=json.loads(Path(a.saved_response).read_text())
        prompt, response=record['prompt_ids'],record['response_ids']
        if not prompt or not response or len(prompt)+len(response)>32768:
            raise ValueError('Require a complete in-budget recorded assistant response')
        ids=torch.tensor([prompt+response],device='cuda')
        selected=torch.arange(len(prompt)-1,ids.shape[1]-1,device='cuda')
        trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if not trainable or any('lora_' not in n or p.dtype!=torch.float32 for n,p in trainable):
            raise RuntimeError('Expected only FP32 LoRA masters')
        opt=torch.optim.AdamW([p for _,p in trainable],lr=1e-6,weight_decay=0.)
        checks=[]
        for iteration in range(2):
            opt.zero_grad(set_to_none=True)
            # Mixed real/padding ranks exercise identical collective ordering.
            padding=iteration==0 and dist.get_rank()==1
            positions=selected[:0] if padding else selected
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,
                    gemma_rl_positions=positions,gemma_rl_labels=ids[0,positions+1])
                loss=-out.gemma_log_probs.sum()/max(1,selected.numel())
            loss.backward()
            grads=[p.grad.to_local() for _,p in trainable if p.grad is not None]
            if len(grads)!=len(trainable) or any(g.dtype!=torch.float32 or not torch.isfinite(g).all() for g in grads):
                raise RuntimeError('Missing or nonfinite FP32 distributed gradients')
            norm=fsdp2_clip_grad_norm_([p for _,p in trainable],1.)
            before=[p.to_local().detach().clone() for _,p in trainable]
            opt.step()
            changed=sum(bool((p.to_local()!=old).any()) for (_,p),old in zip(trainable,before))
            if not changed or not torch.isfinite(norm.full_tensor()):
                raise RuntimeError('Distributed optimizer update missing or nonfinite')
            checks.append({'padding':padding,'loss':loss.item(),'gradient_tensors':len(grads),
                'changed_adapter_tensors':changed,'gradient_norm':norm.full_tensor().item()})
        result={'status':'pass','world_size':2,'rank':dist.get_rank(),'discarded_optimizer_updates':2,
            'production_optimizer_updates':0,'checks':checks,'seconds':time.monotonic()-start,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'saved_response_sha256':hashlib.sha256(Path(a.saved_response).read_bytes()).hexdigest(),
            'adapter_sha256':hashlib.sha256((Path(a.adapter)/'adapter_model.safetensors').read_bytes()).hexdigest()}
        with open(a.output+f'.rank{dist.get_rank()}.json','x') as f:json.dump(result,f,indent=2)
        print(json.dumps(result),flush=True)
        dist.destroy_process_group()
        return
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
