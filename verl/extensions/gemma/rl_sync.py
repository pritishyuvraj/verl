"""Adapter-only transfer contract for the frozen packed Gemma RL base."""
def enabled(model_config):
    if not getattr(model_config, "rl_head_chunk_size", 0):
        return False
    if model_config.weight_loader != "verl.extensions.gemma.rl_head.load_model":
        raise ValueError("Packed RL sync requires the qualified packed loader")
    if model_config.lora_rank <= 0 or model_config.lora.get("merge", False):
        raise ValueError("Packed RL sync transfers unmerged LoRA adapters only")
    return True


def weights(engine):
    values, config = engine.get_per_tensor_param(base_sync_done=True)
    if config is None:
        raise RuntimeError("Adapter metadata is missing")
    for name, value in values:
        if ".lora_A." not in name and ".lora_B." not in name:
            raise RuntimeError(f"Frozen base tensor entered adapter sync: {name}")
        if not value.is_floating_point():
            raise RuntimeError("Packed base tensor entered adapter sync")
        yield name, value


def receiver_kwargs(model_config):
    from peft import PeftConfig
    if not enabled(model_config):
        return {}
    return {"peft_config": PeftConfig.from_pretrained(model_config.lora_adapter_path).to_dict(),
            "base_sync_done": True}
