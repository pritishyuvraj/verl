"""Exact masked-SFT LM-head projection for true-length microbatches of one.

Every context position still runs through Gemma's decoder and attention. Only
positions predicting supervised next tokens are projected to the vocabulary.
The stock verl loss, global normalization and optimizer remain unchanged.
"""

from __future__ import annotations

from types import MethodType


def selected_predecessors(loss_mask):
    import torch

    if loss_mask.ndim != 1 or loss_mask.numel() < 2 or int(loss_mask[0]) != 0:
        raise ValueError("Use one true-length sequence with a masked first token")
    positions = torch.nonzero(torch.roll(loss_mask, shifts=-1) != 0, as_tuple=False).flatten()
    # A neutral row must retain a graph through every decoder layer/FSDP hook.
    # Its single dummy log probability is still multiplied by zero downstream.
    return positions if positions.numel() else torch.zeros(1, dtype=torch.long, device=loss_mask.device)


def scatter_selected_logprobs(logits, full_labels, positions, logprob_function):
    import torch

    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != positions.numel():
        raise ValueError("The selected LM head must return [1, selected_positions, vocabulary]")
    selected = logprob_function(logits.squeeze(0), full_labels.index_select(0, positions), inplace_backward=True)
    return torch.zeros_like(full_labels, dtype=selected.dtype).scatter(0, positions, selected)


def _prepare_inputs(self, micro_batch):
    from verl.utils import tensordict_utils as tu

    if int(micro_batch.batch_size[0]) != 1:
        raise ValueError("Selected-head SFT is qualified only for microbatch size one")
    unsupported = (
        "use_remove_padding",
        "use_fused_kernels",
        "calculate_entropy",
        "calculate_sum_pi_squared",
        "distillation_use_topk",
        "distillation_only",
        "score_centering",
    )
    if any(tu.get_non_tensor_data(data=micro_batch, key=key, default=False) for key in unsupported):
        raise ValueError("Selected-head SFT requires the qualified unfused/no-packing text-only path")
    model_inputs, output_args = self._gemma_dense_prepare_inputs(micro_batch=micro_batch)
    temperature = output_args["temperature"]
    if hasattr(temperature, "numel"):
        if not bool((temperature == 1).all()):
            raise ValueError("Selected-head SFT requires unit temperature")
    elif temperature != 1:
        raise ValueError("Selected-head SFT requires unit temperature")
    ids, mask = micro_batch["input_ids"], micro_batch["loss_mask"]
    positions = selected_predecessors(mask.values())
    model_inputs["logits_to_keep"] = positions
    output_args["gemma_selected_positions"] = positions
    output_args["gemma_full_offsets"] = ids.offsets()
    return model_inputs, output_args


def _prepare_outputs(self, output, output_args, micro_batch, logits_processor_func):
    import functools

    import torch

    from verl.utils.torch_functional import logprobs_from_logits
    from verl.workers.utils.losses import sft_loss

    criterion = logits_processor_func
    while isinstance(criterion, functools.partial):
        criterion = criterion.func
    if criterion is not None and criterion is not sft_loss:
        raise ValueError("Selected-head hooks accept only the stock SFT loss, not RL/distillation processors")

    positions = output_args["gemma_selected_positions"]
    if not torch.is_grad_enabled() and getattr(self, "_gemma_validation_entropy", False):
        mask = torch.roll(micro_batch["loss_mask"].values(), shifts=-1).index_select(0, positions).bool()
        totals = entropy_totals(output.logits.squeeze(0), mask)
        previous = getattr(self, "_gemma_entropy_totals", None)
        self._gemma_entropy_totals = totals if previous is None else previous + totals
    log_probs = scatter_selected_logprobs(
        output.logits, output_args["input_ids_rmpad_rolled"], positions, logprobs_from_logits
    )
    return {"log_probs": torch.nested.nested_tensor_from_jagged(log_probs, output_args["gemma_full_offsets"])}


def entropy_totals(logits, mask, chunk_rows=32):
    """Full-vocabulary entropy over supervised positions, in nats; no loss term."""
    import torch

    with torch.no_grad():
        result = torch.zeros(2, dtype=torch.float64, device=logits.device)
        for start in range(0, logits.shape[0], chunk_rows):
            selected = mask[start : start + chunk_rows]
            if not selected.any():
                continue
            logp = torch.log_softmax(logits[start : start + chunk_rows][selected].float(), dim=-1)
            result[0] += -(logp.exp() * logp).sum(dim=-1).double().sum()
            result[1] += selected.sum()
        return result


def _fp32_head_forward(self, hidden):
    import torch

    # Keep nn.Module.__call__/FSDP's normal pre-forward unshard hooks. Cast the
    # current unsharded frozen tensor; never replace/mutate its Parameter.
    if self.weight.requires_grad or self.bias is not None and self.bias.requires_grad:
        raise ValueError("FP32 head projection is qualified only for a frozen vocabulary head")
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        weight = self.weight.float()
        bias = None if self.bias is None else self.bias.float()
        # Use the identical M=1 GEMM shape for every token in both the dense
        # reference and selected path, including dHidden in backward. Tiny
        # shape-dependent FP32 GEMM differences otherwise cross BF16 rounding
        # boundaries and amplify while traversing the BF16 decoder backward.
        # Cast the frozen weight ONCE; every row's autograd node shares it.
        rows = hidden.float().split(1, dim=-2)
        return torch.cat([torch.nn.functional.linear(row, weight, bias) for row in rows], dim=-2)


def install_fp32_lm_head(model):
    head = model.get_output_embeddings()
    if head is None or not hasattr(head, "weight"):
        raise ValueError("Expected the ordinary Gemma vocabulary projection")
    if hasattr(head, "_gemma_original_head_forward"):
        raise ValueError("FP32 vocabulary-head hook was already installed")
    if head.weight.requires_grad or head.bias is not None and head.bias.requires_grad:
        raise ValueError("The vocabulary projection must remain frozen")
    head._gemma_original_head_forward = head.forward
    head.forward = MethodType(_fp32_head_forward, head)


def install_selected_head(engine, fp32_lm_head=False):
    if hasattr(engine, "_gemma_dense_prepare_inputs"):
        raise ValueError("Selected-head hooks were already installed")
    if getattr(engine, "ulysses_sequence_parallel_size", 1) != 1:
        raise ValueError("Selected-head SFT currently requires sequence-parallel size one")
    engine._gemma_dense_prepare_inputs = engine.prepare_model_inputs
    engine._gemma_dense_prepare_outputs = engine.prepare_model_outputs
    engine.prepare_model_inputs = MethodType(_prepare_inputs, engine)
    engine.prepare_model_outputs = MethodType(_prepare_outputs, engine)
    if fp32_lm_head:
        install_fp32_lm_head(engine.module)
