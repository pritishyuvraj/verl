"""Response-only, checkpointed full-vocabulary Gemma policy statistics.

Every context token passes through the decoder. Projection is restricted to
positions predicting actual sampled assistant tokens; no token is truncated.
"""
from types import MethodType, SimpleNamespace

import torch
from torch.utils.checkpoint import checkpoint


def chunk_statistics(head, hidden, labels, *, chunk_size=32, temperature=1.0, softcap=None):
    if chunk_size < 1 or temperature <= 0:
        raise ValueError("Positive chunk size and temperature are required")
    if labels.numel() == 0:
        raise ValueError("A policy sample must contain generated tokens")

    def project(h, y):
        logits = head(h)
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        logp = torch.log_softmax(logits.float() / temperature, dim=-1)
        chosen = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        entropy = -(logp.exp() * logp).sum(-1)
        return chosen, entropy

    values = []
    for h, y in zip(hidden.split(chunk_size), labels.split(chunk_size), strict=True):
        values.append(checkpoint(project, h, y, use_reentrant=False) if torch.is_grad_enabled() else project(h, y))
    if not values:
        raise ValueError("A policy sample must contain generated tokens")
    return tuple(torch.cat([x[i] for x in values]) for i in range(2))


def _forward(self, *args, gemma_rl_positions=None, gemma_rl_labels=None, gemma_rl_temperature=1.0, **kwargs):
    if gemma_rl_positions is None:
        return self._gemma_original_rl_forward(*args, **kwargs)
    if args or kwargs.get("labels") is not None:
        raise ValueError("RL forward requires named inputs and explicit sampled-token labels")
    kwargs.pop("logits_to_keep", None)
    kwargs.pop("return_dict", None)
    outputs = self.model(**kwargs, return_dict=True)
    hidden = outputs.last_hidden_state
    if hidden.shape[0] != 1:
        raise ValueError("Gemma RL head requires microbatch size one")
    hidden = hidden[0].index_select(0, gemma_rl_positions)
    logp, entropy = chunk_statistics(
        self.lm_head, hidden, gemma_rl_labels,
        chunk_size=self._gemma_rl_chunk_size,
        temperature=gemma_rl_temperature,
        softcap=self.config.get_text_config().final_logit_softcapping,
    )
    return SimpleNamespace(gemma_log_probs=logp, gemma_entropy=entropy)


def load_model(*, auto_class, model_config, dtype):
    from .packed_base import load_model as load_packed
    model = load_packed(auto_class=auto_class, model_config=model_config, dtype=dtype)
    if model.config.model_type != "gemma4":
        raise ValueError("This policy head is qualified only for Gemma4")
    model._gemma_rl_chunk_size = model_config.rl_head_chunk_size
    model._gemma_original_rl_forward = model.forward
    model.forward = MethodType(_forward, model)
    return model


def prepare_inputs(micro_batch, model_inputs, output_args):
    if micro_batch.batch_size[0] != 1:
        raise ValueError("Gemma RL requires one unpadded episode turn per microbatch")
    sequence = micro_batch["input_ids"].values()
    response = micro_batch["responses"].values()
    prompt_length = sequence.numel() - response.numel()
    if prompt_length < 1 or not torch.equal(sequence[prompt_length:], response):
        raise ValueError("Input/response token alignment mismatch")
    mask = micro_batch["response_mask"].values().bool()
    selected = torch.where(mask)[0]
    positions = selected + prompt_length - 1
    temperature = output_args["temperature"].flatten()
    if temperature.numel() != 1:
        raise ValueError("One sampling temperature is required per turn")
    model_inputs.update(gemma_rl_positions=positions, gemma_rl_labels=response[selected],
                        gemma_rl_temperature=float(temperature.item()))
    return positions


def restore_outputs(raw, micro_batch, positions):
    ids = micro_batch["input_ids"]
    return {
        name: torch.nested.nested_tensor_from_jagged(
            value.new_zeros(ids.values().numel()).index_copy(0, positions, value), ids.offsets())
        for name, value in (("log_probs", raw.gemma_log_probs), ("entropy", raw.gemma_entropy))
    }
